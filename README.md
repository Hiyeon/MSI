
# Self-Supervised Medical Slice Interpolation Network Using Controllable Feature Flow

### Authors: Pengcheng Lei, Faming Fang, Tingting Wang, Cong Liu, Guixu Zhang

### abstract
Deep learning-based image interpolation methods are confronted with various challenges in its application to anisotropic medical
volumetric data (i.e., CT and MR images) out of the complex nonlinear deformation and the scarcity of high-quality images.
This paper proposes a self-supervised multiple medical slice interpolation network using controllable feature flow. Firstly, a
Controllable Feature Flow Network (CFFNet) is proposed to estimate the complex nonlinear deformation of medical images. In
CFFNet, we first use a deformation-aware network and a spatial channel modulation module to predict the bi-directional feature
flows from the source slices to the target slices by considering an additional position parameter, and then the learned feature
flows are used to synthesize the target intermediate features by using the deformable convolution. Secondly, an improved two-stage self-supervised framework is proposed to alleviate the dependency on high-quality medical images. In the first stage, the
synthesized training pairs along the dense sagittal and coronal directions are adopted to pre-train the CFFNet. In the second one,
the sparse axial slices are used to fine-tune the CFFNet with the cycle-consistency constraint and a feature domain smooth loss.
Experimental results illustrate that the proposed CFFNet assumes superior performance on medical slice interpolation tasks with
fewer parameters and the proposed self-supervised CFFNet obtains competitive results compared with other fully supervised
algorithms. We also evaluate our model on HepaticVessel segmentation task, the proposed method can effectively enhance the
continuity and smoothness of the 3D vessel structures, showing promising results on clinical application.

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


