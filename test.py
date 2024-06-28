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

import h5py
import torch.nn.functional as F

def load_specific_index(file_path, index):
    with h5py.File(file_path, 'r') as f:
        # 특정 인덱스의 데이터만 로드
        img_LQ = f['LQs'][index]
        img_GT = f['GT'][index]
    return img_LQ, img_GT

def get_dataset_length(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        # 데이터셋의 길이(첫 번째 차원의 크기)를 반환
        length = f[dataset_name].shape[0]
    return length

def check_if_folder_exist(folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

def test():
    code_name = 'SMWI_2mmto1mm'
    data_mode = 'Frame_testAx' #'trainSagCor_testAx'
    result_folder = './results'
    N_ot = 1     # control the slice number you want to interpolate [1, 3, 5]
    use_time = True
    save_imgs = True   # wheather save test images
    is_trilinear = True

    # pre-trained model path
    model_path = './experiments/SMWI_2mmto1mm_Frame/models/Iter_8000.pth'

    # test file path
    dataset_file = './dataset/test_data_ax.h5'

    model = OURS_new.CFFNet(front_RBs=5)
    
    #### dataset    
    # folder = os.path.join(result_folder, data_mode)
    save_folder = os.path.join(result_folder, data_mode, code_name)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    util.mkdirs(save_folder)
    
    util.setup_logger(logger_name=code_name + '_with_' + data_mode, root=save_folder, phase=code_name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger(name=code_name + '_with_' + data_mode)
    model_params = util.get_model_total_params(model)


    #### log info
    logger.info('Data: {} - {}'.format(data_mode, dataset_file))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))


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

            # if 'OURS' in code_name:
            model_output = model(imgs_in, time_Tensors) 
            # else:   
            #     model_output = model(imgs_in)     
            output = model_output
            
            return output

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    data_len = get_dataset_length(dataset_file, 'GT')
    
    avg_psnr_sum, cal_n = 0,0
    avg_ssim_sum, cal_n = 0,0

    if is_trilinear:
        ref_psnr_sum, ref_ssim_sum = 0,0

    for i in range(data_len):
        # get input images
        img_LQs, img_GT = load_specific_index(dataset_file, i)

        img_GT = np.float32(img_GT)
        img_GT = np.transpose(img_GT, (1,2,0)) # (c, h, w)

        imgs_in = torch.from_numpy(np.float32(img_LQs)).unsqueeze(1).unsqueeze(0).to(device) #(1, 2, 1, w, h)
        output = single_forward(model, imgs_in, use_time=use_time, N_ot=N_ot)  # (1, 1, 1, w, h)

        output = np.array(output.cpu().squeeze(0).squeeze(0)) # (c, h, w)
        output = np.transpose(output, (1,2,0)) # (h, w, c)

        if is_trilinear:
            tmp_in = torch.from_numpy(np.float32(img_LQs)).unsqueeze(0).unsqueeze(0) # (b, c, d, h, w)
            b,c,d,h,w = tmp_in.shape
            img_fake = F.interpolate(tmp_in, size=(3, h, w), mode='trilinear', align_corners=False)
            img_fake = np.transpose(img_fake.squeeze(0).squeeze(0), (1,2,0)) # (h,w,d)
            img_fake = np.array(img_fake[:,:,1].unsqueeze(-1))

        if save_imgs:
            # save imgs
            cv2.imwrite(osp.join(save_folder, '{:08d}_gt.png'.format(i+1)), img_GT)
            cv2.imwrite(osp.join(save_folder, '{:08d}_out.png'.format(i+1)), output)
            if is_trilinear:
                cv2.imwrite(osp.join(save_folder, '{:08d}_tri.png'.format(i+1)), img_fake)

        crt_psnr = util.calculate_psnr(output, img_GT)
        crt_ssim = util.calculate_ssim(output, img_GT)
        logger.info('{:3d}-th image \tPSNR: {:.6f} dB \tSSIM: {:.6f} dB'.format(i+1, crt_psnr, crt_ssim))
        if crt_ssim != 1:
            avg_psnr_sum += crt_psnr
            avg_ssim_sum += crt_ssim
            if is_trilinear:
                ref_psnr_sum += util.calculate_psnr(img_fake, img_GT)
                ref_ssim_sum += util.calculate_ssim(img_fake, img_GT)

            cal_n += 1

    avg_psnr = avg_psnr_sum / cal_n
    avg_ssim = avg_ssim_sum / cal_n

    if is_trilinear:
        ref_psnr = ref_psnr_sum / cal_n
        ref_ssim = ref_ssim_sum / cal_n

    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, dataset_file))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. Total Average SSIM: {:.6f} dB for {} clips.'.format(avg_psnr, cal_n, avg_ssim, cal_n))
    if is_trilinear:
        logger.info('Total Trilinear PSNR: {:.6f} dB for {} clips. Total Trilinear SSIM: {:.6f} dB for {} clips.'.format(ref_psnr, cal_n, ref_ssim, cal_n))


### CUDA_VISIBLE_DEVICES=0 python test_multi_frame_psnr.py
if __name__ == '__main__':
    test()
