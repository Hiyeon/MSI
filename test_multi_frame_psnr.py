import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import utils.util as util
import data.util as data_util
import models.modules.OURS_new as OURS_new
import csv
import time
import skimage
import skimage.metrics as sm
import itertools
import re
import options.options as option

def check_if_folder_exist(folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

def test():
    code_name = 'OURS_x2'
    data_mode = 'temp'
    result_folder = '/fast_storage/hayeon/MSI/code/results'
    N_ot = 1     # control the slice number you want to interpolate [1, 3, 5]
    use_time = True
    save_imgs = True   # wheather save test images

    # pre-trained model path
    model_path = './check_points/x2.pth'
    # model_path = './check_points/x4.pth'
    # model_path = './check_points/x6.pth'

    # test file path
    dataset_folder =  '../data/LITS_volume_img/*'
    # dataset_folder = '/data/vessel_volume_img/*'
    # dataset_folder = '/data/kidney_volume_img/*'
    # dataset_folder = '/data/colon_volume_img/*'

    model = OURS_new.CFFNet(front_RBs=5)
    
    #### dataset    
    test_dataset_folder = dataset_folder  
    folder = os.path.join(result_folder, data_mode)
    save_folder = os.path.join(folder, code_name)
    save_visualization_folder = os.path.join(folder, code_name)

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')

    util.mkdirs(save_folder)
    
    util.setup_logger(logger_name=code_name + '_with_' + data_mode, root=folder, phase=code_name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger(name=code_name + '_with_' + data_mode)
    model_params = util.get_model_total_params(model)


    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def single_forward(model, imgs_in, use_time=None, N_ot=None):
        with torch.no_grad():
            b,n,c,h,w = imgs_in.size()  #[1,2,1,h,w]
            if use_time == True:
                if N_ot==1:
                    time_number = [[0.5]]
                elif N_ot==2:
                    time_number = [[0.25, 0.75]]
                elif N_ot==3:
                    time_number = [[0.25, 0.5, 0.75]]
                elif N_ot==5:
                    time_number = [[1/6, 2/6, 3/6, 4/6, 5/6]]
                time_Tensors = torch.Tensor(time_number)
                time_Tensors = time_Tensors.to(device)
            else:
                time_Tensors = None

            if 'OURS' in code_name:
                model_output = model(imgs_in, time_Tensors) 
            else:   
                model_output = model(imgs_in)     
            output = model_output
            
            return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_ssim_l = []
    sub_folder_name_l = []

    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)
        save_visualization_sub_folder = osp.join(save_visualization_folder, sub_folder_name)

        img_LR_l = sorted(glob.glob(sub_folder + '/*'))

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR seq images
        imgs = util.read_seq_imgs(sub_folder)
        #### read GT seq images
        img_GT_l = []
        
        sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/HR/'))

        if 'Custom' not in data_mode:
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*'))):
                img_GT_l.append(util.read_image(img_GT_path))

        avg_psnr, avg_psnr_sum, cal_n = 0,0,0
        avg_ssim, avg_ssim_sum, cal_n = 0,0,0
        
        #[[[0, 2], [0, 1, 2]], [[2, 4], [2, 3, 4]] ...] 
        skip = True
        select_idx_list = util.test_index_generation(skip, N_ot+2, len(img_LR_l))
        # print(select_idx_list)
        if len(img_LR_l) > 3:
            select_idx_list.append([[len(img_LR_l)-3,len(img_LR_l)-1], [len(img_LR_l)-3, len(img_LR_l)-2, len(img_LR_l)-1]])

       # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = [select_idxs[0][0], select_idxs[0][-1]]  #取首位两张图片输入 (첫 번째 두 장의 그림을 취하여 입력)
            gt_idx = select_idxs[1][1:-1]  # 中间图片为gt (가운데 사진은 gt)

            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device) #(1, 2, 1, w, h)
            output = single_forward(model, imgs_in, use_time=use_time, N_ot=N_ot)  # (1, 3, 1, w, h)

            outputs = []
            for img in range(N_ot):
                outputs.append(output[:, img, :, :, :])
            outputs = torch.stack(outputs, 1)
            outputs = outputs.data.float().cpu().squeeze(0)   #(n, c, h, w)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            ref_img = util.tensor2img(imgs_in[0, 0, 0, :, :])
            cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(select_idxs[0][0]+1)), ref_img)
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:, :].squeeze(0)

                output = util.tensor2img(output_f)
                if save_imgs:
                    cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1)), output)

                if 'Custom' not in data_mode:
                    #### calculate PSNR
                    output = output / 255.
                    GT = np.copy(img_GT_l[name_idx])

                    # GT转为单通道
                    GT = GT[:, :, 0]

                    crt_psnr = util.calculate_psnr(output * 255., GT * 255.)
                    crt_ssim = util.calculate_ssim(output * 255., GT * 255.)
                    logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB \tSSIM: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_ssim))
                    if crt_ssim != 1:
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        cal_n += 1

        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_ssim = avg_ssim_sum / cal_n
    
            logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; - Average SSIM: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, cal_n, avg_ssim, cal_n))

            avg_psnr_l.append(avg_psnr)
            avg_ssim_l.append(avg_ssim)

    if 'Custom' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, ssim in zip(sub_folder_name_l, avg_psnr_l, avg_ssim_l):
            logger.info('Folder {} - Average PSNR: {:.6f} dB - Average SSIM: {:.6f} dB. '.format(name, psnr, ssim))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip Test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB for {} clips. Total Average SSIM: {:.6f} dB for {} clips.'.format(sum(avg_psnr_l) / len(avg_psnr_l), len(sub_folder_l), sum(avg_ssim_l) / len(avg_ssim_l), len(sub_folder_l)))



### CUDA_VISIBLE_DEVICES=0 python test_multi_frame_psnr.py
if __name__ == '__main__':
    test()
