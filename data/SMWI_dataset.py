'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random

import logging
import numpy as np

import torch
import torch.utils.data as data
import data.util as util
import os

import pydicom

def load_dicom_folder(folder_path):
    # List all files in the folder
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort()
    
    # Read and sort the DICOM files by Instance Number
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicoms.sort(key=lambda x: int(x.InstanceNumber))

    # Stack the pixel data into a 3D array
    image_stack = np.stack([d.pixel_array.astype(np.float64) for d in dicoms])

    return image_stack

def wins(inp):
    # Define the percentiles
    b = [0.5, 99.5]
    # Calculate the lower and upper bounds based on percentiles
    lb = np.percentile(inp, np.min(b))
    ub = np.percentile(inp, np.max(b))
    # Copy input array to avoid modifying it directly
    y = np.copy(inp)
    # Apply limits
    y[y < lb] = lb
    y[y > ub] = ub
    # Normalize to [0, 1]
    if ub == lb:
        out = np.zeros_like(y)  # 모든 값을 0으로 설정
    else:
        out = (y - lb) / (ub - lb)
    return out

logger = logging.getLogger('base')

class SMWIDataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    GT: Ground-Truth;
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(SMWIDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        self.LR_index_list = []
        for i in range(self.LR_N_frames):
            self.LR_index_list.append(i*2)

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        GT_list = sorted(os.listdir(self.GT_root))
        self.paths_GT = [os.path.join(self.GT_root, i) for i in GT_list]

        assert self.paths_GT, 'Error: GT path is empty.'

        self.subject_num = len(self.paths_GT)

        vol_ = load_dicom_folder(self.paths_GT[0])
        if opt['stage1']:
            vol_ = np.transpose(vol_, (2,1,0))
        self.slice_num = len(vol_)
        self.data_size = ( self.slice_num - opt['N_frames'] ) * self.subject_num

        if self.data_type == 'dicom':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))


    def __getitem__(self, index):
        N_frames = self.opt['N_frames']
        GT_size = self.opt['GT_size']

        #### get the GT & LQ image (as the center frame)
        subject_index = index // ( self.slice_num - N_frames )
        slice_index = index % ( self.slice_num - N_frames )

        img_GT_l = []
        img_LQ_l = []

        vol_GT = load_dicom_folder(self.paths_GT[subject_index]) # (32, 384, 384)
        vol_GT = wins(vol_GT)*255.0
        # vol_GT = vol_GT[2:-2,:,:] # (28, 384, 384) -> remove zeropadding (SMWI)

        if self.opt['stage1']:
            sag_GTs = np.transpose(vol_GT, (2,1,0)) # DHW -> WHD
            cor_GTs = np.transpose(vol_GT, (1,2,0)) # DHW -> HWD

            # sagittal slices
            sag_ = sag_GTs[slice_index:slice_index+N_frames,:,:] # (N_frames, H, W)
            sag_GT = np.expand_dims(sag_[1:-1,:,:], axis=-1) # (N_frames-2, H, W, C=1)
            sag_LQ = np.expand_dims(np.stack([sag_[0,:,:], sag_[-1,:,:]], axis=0), axis=-1) # (2, H, W, C=1)

            img_GT_l.append(sag_GT)
            img_LQ_l.append(sag_LQ)
        
            # coronal slices
            cor_ = cor_GTs[slice_index:slice_index+N_frames,:,:]
            cor_GT = np.expand_dims(cor_[1:-1,:,:], axis=-1) # (N_frames-2, H, W, C=1)
            cor_LQ = np.expand_dims(np.stack([cor_[0,:,:], cor_[-1,:,:]], axis=0), axis=-1) # (2, H, W, C=1)

            img_GT_l.append(cor_GT)
            img_LQ_l.append(cor_LQ)

        else:
            # axial slices
            img_ = vol_GT[slice_index:slice_index+N_frames,:,:] # (N_frames, 384, 384)
            img_GT = img_[1:-1,:,:] # (N_frames-2, 384, 384)
            img_LQ = np.stack([img_[0,:,:], img_[-1,:,:]], axis=0) # (2, 384, 384)
            
            img_GT = np.expand_dims(img_GT, axis=-1) # (N_frames-2, 384, 384, 1)
            img_LQ = np.expand_dims(img_LQ, axis=-1) # (2, 384, 384, 1)

            img_GT_l.append(img_GT)
            img_LQ_l.append(img_LQ)
    
        if self.opt['phase'] == 'train':
            if self.opt['stage1']: # relatively narrow w in sag and cor slices
                B, H, W, C = img_LQ_l[0].shape
                rnd_h = random.randint(0, max(0, H - GT_size))
                GT_size_ = min(W, GT_size)
                rnd_w = random.randint(0, max(0, W - GT_size_))
                img_LQ_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size_, :] for v in img_LQ_l]
                img_GT_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size_, :] for v in img_GT_l]
            else:
                B, H, W, C = img_LQ_l[0].shape
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_GT_l]

            # augmentation - flip, rotate
            img_LQ_l = img_LQ_l + img_GT_l
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-len(rlt)//2]
            img_GT_l = rlt[-len(rlt)//2:]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0) # BNHWC
        img_GTs = np.stack(img_GT_l, axis=0)

        # 转为单通道 (단일 채널로 전환)
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 1, 4, 2, 3)))).float() # BNCHW
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 1, 4, 2, 3)))).float()

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
        
        time_tensor = time_tensor.expand(len(img_GTs), 1) # B, 1
        
        if self.opt['ssl'] and self.opt['use_time'] == True:
            return {'LQs': img_GTs, 'GT': img_GTs, 'time': time_tensor}

        elif self.opt['ssl']:
            return {'LQs': img_GTs, 'GT': img_GTs}
        
        elif self.opt['use_time'] == True:
            return {'LQs': img_LQs, 'GT': img_GTs, 'time': time_tensor}
        
        else:
            return {'LQs': img_LQs, 'GT': img_GTs}

    def __len__(self):
        return self.data_size
