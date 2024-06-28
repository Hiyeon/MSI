import logging
import numpy as np

import torch
import torch.utils.data as data
import os
import h5py

import random

def get_dataset_length(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        length = f[dataset_name].shape[0]
    return length

def load_specific_index(file_path, index):
    with h5py.File(file_path, 'r') as f:
        # 특정 인덱스의 데이터만 로드
        img_LQ = f['LQs'][index]
        img_GT = f['GT'][index]
    return img_LQ, img_GT

logger = logging.getLogger('base')

class SMWI_h5_Dataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    GT: Ground-Truth;
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(SMWI_h5_Dataset, self).__init__()
        self.opt = opt

        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        self.LR_index_list = []
        for i in range(self.LR_N_frames):
            self.LR_index_list.append(i*2)

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']

        if self.GT_root.endswith('.h5'):
            pass
        else:
            _, ext = os.path.splitext(self.GT_root)
            raise ValueError('Wrong data type: {}'.format(ext))


    def __getitem__(self, index):
        img_LQ, img_GT = load_specific_index(self.GT_root, index)

        # print("min, max of img_LQ:", np.min(img_LQ), np.max(img_LQ))
        # print("min, max of img_GT:", np.min(img_GT), np.max(img_GT))

        # N,H,W,C=1
        img_GT = np.expand_dims(img_GT, axis=-1)
        img_LQ = np.expand_dims(img_LQ, axis=-1)

        # 단일 채널로 전환
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (0, 3, 1, 2)))).float() # NCHW
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (0, 3, 1, 2)))).float()

        if self.opt['is_crop']:
            N, C, H, W = img_GTs.shape
            
            GT_size = self.opt['GT_size']
            H_size = min(H, GT_size)
            W_size = min(W, GT_size)
            
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQs = img_LQs[:, :, rnd_h:rnd_h + H_size, rnd_w:rnd_w + W_size]
            img_GTs = img_GTs[:, :, rnd_h:rnd_h + H_size, rnd_w:rnd_w + W_size]

        time_list = []
        for i in range(5):
            time_list.append(torch.Tensor([i / (5 - 1)]))
        time_Tensors = torch.cat(time_list)
        if self.opt['N_frames']==5:
            time_tensor = time_Tensors[[1, 2, 3]]
        elif self.opt['N_frames']==3:
            time_tensor = time_Tensors[[2]]
        elif self.opt['N_frames']==7:
            time_tensor = torch.tensor([1/6, 2/6, 3/6, 4/6, 5/6])
        
        if self.opt['ssl'] and self.opt['use_time'] == True:
            return {'LQs': img_GTs, 'GT': img_GTs, 'time': time_tensor}

        elif self.opt['ssl']:
            return {'LQs': img_GTs, 'GT': img_GTs}
        
        elif self.opt['use_time'] == True:
            return {'LQs': img_LQs, 'GT': img_GTs, 'time': time_tensor}
        
        else:
            return {'LQs': img_LQs, 'GT': img_GTs}

    def __len__(self):
        return get_dataset_length(self.GT_root, 'GT')
