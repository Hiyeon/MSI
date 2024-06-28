import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

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

class Val_SMWIdataset(data.Dataset):

    def __init__(self, opt):
        super(Val_SMWIdataset).__init__()
        self.opt = opt
        self.n_frame = self.opt["N_frames"]

        GT_rootpath = self.opt['dataroot_GT']
        GT_list = sorted(os.listdir(GT_rootpath))
        self.paths_GT = [os.path.join(GT_rootpath, i) for i in GT_list]

        LQ_rootpath = self.opt['dataroot_LQ']
        LQ_list = sorted(os.listdir(LQ_rootpath))
        self.paths_LQ = [os.path.join(LQ_rootpath, i) for i in LQ_list]

        # self.subject_num = len(self.paths_GT)

        # vol_ = load_dicom_folder(self.paths_GT[0])
        # if self.opt['stage1']:
        #     vol_ = np.transpose(vol_, (2,1,0))
        # self.slice_num = len(vol_)
        # self.data_size = ( self.slice_num - self.n_frame ) * self.subject_num

    def __getitem__(self, index):
        img_GT_l = []
        img_LQ_l = []

        vol_GT = load_dicom_folder(self.paths_GT[index]) # (32, 384, 384)
        vol_GT = wins(vol_GT)*255.0

        if self.opt['stage1']:
            sag_GTs = np.transpose(vol_GT, (2,1,0)) # DHW -> WHD
            cor_GTs = np.transpose(vol_GT, (1,2,0)) # DHW -> HWD

            # sagittal slices
            N_sampling = (len(sag_GTs)-1)//(self.n_frame-1)
            for ind in range(N_sampling):
                ind = ind*(self.n_frame-1)
                sag_ = sag_GTs[ind:ind+self.n_frame,:,:] # (N_frames, H, W)
                sag_GT = np.expand_dims(sag_[1:-1,:,:], axis=-1) # (N_frames-2, H, W, C=1)
                sag_LQ = np.expand_dims(np.stack([sag_[0,:,:], sag_[-1,:,:]], axis=0), axis=-1) # (2, H, W, C=1)
                if np.all(sag_LQ == 0):
                    pass
                else:
                    img_GT_l.append(sag_GT)
                    img_LQ_l.append(sag_LQ)
        
            # coronal slices
            N_sampling = (len(sag_GTs)-1)//(self.n_frame-1)
            for ind in range(N_sampling):
                ind = ind*(self.n_frame-1)
                cor_ = cor_GTs[ind:ind+self.n_frame,:,:]
                cor_GT = np.expand_dims(cor_[1:-1,:,:], axis=-1) # (N_frames-2, H, W, C=1)
                cor_LQ = np.expand_dims(np.stack([cor_[0,:,:], cor_[-1,:,:]], axis=0), axis=-1) # (2, H, W, C=1)
                if np.all(cor_LQ == 0):
                    pass
                else:
                    img_GT_l.append(cor_GT)
                    img_LQ_l.append(cor_LQ)
        else:
            # axial slices
            N_sampling = (len(vol_GT)-1)//(self.n_frame-1)
            for ind in range(N_sampling):
                ind = ind*(self.n_frame-1)
                img_ = vol_GT[ind:ind+self.n_frame,:,:]
                img_GT = np.expand_dims(img_[1:-1,:,:], axis=-1) # (N_frames-2, H, W, C=1)
                img_LQ = np.expand_dims(np.stack([img_[0,:,:], img_[-1,:,:]], axis=0), axis=-1) # (2, H, W, C=1)
                if np.all(img_LQ == 0):
                    pass
                else:
                    img_GT_l.append(img_GT)
                    img_LQ_l.append(img_LQ)

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 1, 4, 2, 3)))).float()
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 1, 4, 2, 3)))).float() # BNCHW

        ## batch validation due to memory issue ##
        total_batches = len(img_GTs)
        num_samples = self.opt['sample_batch']

        middle_index = total_batches // 2
        start_index = max(middle_index - num_samples // 2, 0)  # 시작 인덱스가 0 미만이 되지 않도록
        end_index = min(start_index + num_samples, total_batches)  # 끝 인덱스가 전체 배치 수를 넘지 않도록

        img_LQs = img_LQs[start_index:end_index, :, :, :, :]
        img_GTs = img_GTs[start_index:end_index, :, :, :, :]

        time_list = []
        for i in range(5):
            time_list.append(torch.Tensor([i / (5 - 1)]))
        time_Tensors = torch.cat(time_list)
        if self.opt['N_frames']==5:
            time_tensor = time_Tensors[[1, 2, 3]]
        elif self.opt['N_frames']==4:
            time_tensor = time_Tensors[[1, 3]]
        elif self.opt['N_frames']==3:
            time_tensor = time_Tensors[[2]]
        elif self.opt['N_frames']==7:
            time_tensor = torch.tensor([1/6, 2/6, 3/6, 4/6, 5/6])

        time_tensor = time_tensor.expand(len(img_GTs), 1) # B, 1

        if self.opt['use_time'] == True:
            return {'LQs': img_LQs, 'GT': img_GTs, 'time': time_tensor}
        else:
            return {'LQs': img_LQs, 'GT': img_GTs}
        
    def __len__(self):
        return len(self.paths_LQ)