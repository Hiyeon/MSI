import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss, Fourier_loss, GANLoss, VGG_fea
from torchvision import models
import torchvision
import numpy as np

logger = logging.getLogger('base')

class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)

        self.opt = opt
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG_pretrain = networks.define_G(opt).to(self.device)
        # 提取vgg特征 (vgg 피쳐 추출)
        self.vgg19 = VGG_fea().to(self.device)

        if opt['datasets']['train']['use_time'] == True and opt['time_pth'] != None:
            self.netG.load_state_dict(torch.load(opt['time_pth']), strict=True)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            self.netG_pretrain = DistributedDataParallel(self.netG_pretrain, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
            self.netG_pretrain = DataParallel(self.netG_pretrain)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            self.netG_pretrain.eval()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device) # Penalty term(𝜖)을 넣어 Outlier에 robust함
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)

                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():
                # if opt['datasets']['train']['use_time'] == True and 'TMB' not in k:
                #     v.requires_grad = False
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            # if opt['ssl'] == True:
            #     self.optimizer_G = torch.optim.Adam(self.netG.named_parameters(), lr=train_opt['lr_G'],
            #                                     weight_decay=wd_G,
            #                                     betas=(train_opt['beta1'], train_opt['beta2'])) 
            #     self.optimizers.append(self.optimizer_G)
            # else:
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2'])) 
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()
    
    def feed_data(self, data, need_GT=True):
        if 'LQs_01' in data:
            self.var_L_01 = data['LQs_01'].to(self.device) # B, N, C, H, W
            self.var_L_12 = data['LQs_12'].to(self.device)
        else:
            self.var_L = data['LQs'].to(self.device)
        # T向量 (T벡터)：（batch,3）[0.2500, 0.5000, 0.7500]
        if self.opt['use_time'] == True:
            self.var_T = data['time'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0
        self.optimizers[1].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        if self.opt['use_time'] == True:
            self.fake_H_01 = self.netG(self.var_L_01, t=self.var_T)
            self.fake_H_12 = self.netG(self.var_L_12, t=self.var_T)
            self.pre_H_01 = self.netG_pretrain(self.var_L_01, t=self.var_T)
            self.pre_H_12 = self.netG_pretrain(self.var_L_12, t=self.var_T)
            # stack
            self.fake_H_1 = self.netG(torch.cat([self.fake_H_01, self.fake_H_12],dim=1), t=self.var_T)
        else:
            self.fake_H_01 = self.netG(self.var_L_01)
            self.fake_H_12 = self.netG(self.var_L_12)
            self.pre_H_01 = self.netG_pretrain(self.var_L_01)
            self.pre_H_12 = self.netG_pretrain(self.var_L_12)
            # stack
            self.fake_H_1 = self.netG(torch.cat([self.fake_H_01, self.fake_H_12],dim=1))

        self.vgg_fea_01 = self.vgg19(self.fake_H_01, self.pre_H_01)
        self.vgg_fea_12 = self.vgg19(self.fake_H_12, self.pre_H_12)
        
        l_pix = self.cri_pix(self.fake_H_1, self.real_H) + self.opt['beta'] * ( self.cri_pix(self.vgg_fea_01[0], self.vgg_fea_01[1]) + self.cri_pix(self.vgg_fea_12[0], self.vgg_fea_12[1]) ) 
        l_pix.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.opt['use_time'] == True:
                self.fake_H = self.netG(self.var_L, t=self.var_T)
            else:
                self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.float().cpu()
        out_dict['restore'] = self.fake_H.float().cpu()
        # out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        # out_dict['restore'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.float().cpu()
            # out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            self.load_network(load_path_G, self.netG_pretrain, self.opt['path']['strict_load'])

    def save(self, iter_label, epoch=None):
        if epoch != None:
            self.save_network(self.netG, 'G', iter_label, str(epoch))
        else:
            self.save_network(self.netG, 'G', iter_label)
