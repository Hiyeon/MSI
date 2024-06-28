# Self-supervised Medical Slice Interpolation Network Using Controllable Feature Flow

## Description
This repository contains the unofficial implementation of the paper ["Self-supervised Medical Slice Interpolation Network Using Controllable Feature Flow"](https://www.sciencedirect.com/science/article/pii/S0957417423024454) by Lei, published in Expert Systems with Applications (2024). The project aims to enhance medical image processing by using a self-supervised learning approach for interpolating medical slices with controllable feature flow.

Deep learning-based image interpolation methods are confronted with various challenges in their application to anisotropic medical volumetric data (i.e., CT and MR images) due to complex nonlinear deformations and the scarcity of high-quality images. This paper proposes a self-supervised multiple medical slice interpolation network using controllable feature flow.

- **Controllable Feature Flow Network (CFFNet)**: Estimates complex nonlinear deformations in medical images. CFFNet utilizes a deformation-aware network and a spatial channel modulation module to predict bi-directional feature flows from source slices to target slices, considering an additional position parameter. The learned feature flows are then used to synthesize target intermediate features via deformable convolution.
- **Two-stage Self-supervised Framework**: In the first stage, synthesized training pairs along the dense sagittal and coronal directions pre-train the CFFNet. In the second stage, sparse axial slices fine-tune the CFFNet with a cycle-consistency constraint and feature domain smooth loss.

This implementation is based on the original code provided by the authors, which can be found at [https://codeocean.com/capsule/8945297/tree/v1]. I have added the training code and revised the data loading process to improve usability and functionality.

## Key Features
- **Self-supervised Learning**: Utilizes self-supervised techniques to reduce the need for labeled data.
- **Controllable Feature Flow**: Allows for precise control over the interpolation process.

**1.Compiling DCNv2:**
```bash
python ./models/modules/DCNv2/setup.py build develop
```

**2.Model training:**

Modify the data set path and training parameters in [_configs/TMNet_multiple_frames.yaml_], then run
```bash
sh train.sh
```

**3. Fast test:**

Modify the test configurations in Python file _test_multi_frame_psnr.py_. Note that we have uploaded the weights of our models in file _check_points_. 
```bash
result_folder = './results'
N_ot = 1     # control the slice number you want to interpolate. Select [1, 3, 5] for [x2, x4, x6] respectively.
use_time = True   # multi-slice interpolation
save_imgs = False   # wheather save test images

# pre-trained model path
model_path = './check_points/x2.pth'   
# model_path = './check_points/x4.pth'
# model_path = './check_points/x6.pth'

# test file path
dataset_folder =  '../data/LITS_volume_img/*'
# dataset_folder = '/data/vessel_volume_img/*'
# dataset_folder = '/data/kidney_volume_img/*'
# dataset_folder = '/data/colon_volume_img/*'
```
Then run:
```bash
sh test.sh
```
We get the testing results as:
```bash
23-09-23 11:40:04.600 - INFO: Data: temp - /data/LITS_volume_img/*
23-09-23 11:40:04.600 - INFO: Padding mode: replicate
23-09-23 11:40:04.600 - INFO: Model path: ./check_points/x2.pth
23-09-23 11:40:04.600 - INFO: Model parameters: 1.507629 M
23-09-23 11:40:04.600 - INFO: Save images: False
23-09-23 11:40:04.600 - INFO: Flip Test: False
23-09-23 11:40:13.454 - INFO:   2 -                         2.png 	PSNR: 39.060180 dB 	SSIM: 0.972012 dB
23-09-23 11:40:13.828 - INFO:   4 -                         4.png 	PSNR: 39.320951 dB 	SSIM: 0.973429 dB
23-09-23 11:40:14.202 - INFO:   6 -                         6.png 	PSNR: 39.100264 dB 	SSIM: 0.972124 dB
23-09-23 11:40:14.576 - INFO:   8 -                         8.png 	PSNR: 39.151693 dB 	SSIM: 0.972499 dB
23-09-23 11:40:14.949 - INFO:  10 -                        10.png 	PSNR: 39.002313 dB 	SSIM: 0.971758 dB
23-09-23 11:40:15.322 - INFO:  12 -                        12.png 	PSNR: 39.097148 dB 	SSIM: 0.973091 dB
23-09-23 11:40:15.695 - INFO:  14 -                        14.png 	PSNR: 39.057419 dB 	SSIM: 0.973872 dB
... ...
```

## Reference
1. Lei, “Self-supervised medical slice interpolation network using controllable feature flow”, Expert Systems with Applications (2024).


