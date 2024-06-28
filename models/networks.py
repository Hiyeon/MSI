import models.modules.OURS_new as OURS_new

####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'OURS_new':
        netG = OURS_new.CFFNet(nf=opt_net['nf'], nframes=opt_net['nframes'], groups=opt_net['groups'], front_RBs=opt_net['front_RBs'], opt=opt)
    ###########################
    elif which_model == 'sr3':
        from .sr3_modules import diffusion, unet
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
        netG = diffusion.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',    # L1 or L2
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train']
        )
        if opt['phase'] == 'train':
            # init_weights(netG, init_type='kaiming', scale=0.1)
            init_weights(netG, init_type='orthogonal')
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            netG = nn.DataParallel(netG)
    ###########################
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
