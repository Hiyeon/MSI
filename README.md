# Self-supervised Medical Slice Interpolation Network Using Controllable Feature Flow

## Description
This repository contains the unofficial implementation of the paper ["Self-supervised Medical Slice Interpolation Network Using Controllable Feature Flow"](https://www.sciencedirect.com/science/article/pii/S0957417423024454) by Lei, published in Expert Systems with Applications (2024). The project aims to enhance medical image processing by using a self-supervised learning approach for interpolating medical slices with controllable feature flow.

Deep learning-based image interpolation methods are confronted with various challenges in their application to anisotropic medical volumetric data (i.e., CT and MR images) due to complex nonlinear deformations and the scarcity of high-quality images. This paper proposes a self-supervised multiple medical slice interpolation network using controllable feature flow.

- **Controllable Feature Flow Network (CFFNet)**: Estimates complex nonlinear deformations in medical images. CFFNet utilizes a deformation-aware network and a spatial channel modulation module to predict bi-directional feature flows from source slices to target slices, considering an additional position parameter. The learned feature flows are then used to synthesize target intermediate features via deformable convolution.
- **Two-stage Self-supervised Framework**: In the first stage, synthesized training pairs along the dense sagittal and coronal directions pre-train the CFFNet. In the second stage, sparse axial slices fine-tune the CFFNet with a cycle-consistency constraint and feature domain smooth loss.

This implementation is based on the original code provided by the authors, which can be found at https://codeocean.com/capsule/8945297/tree/v1. I have added the training code and revised the data loading process to improve usability and functionality.

## Key Features
- **Self-supervised Learning**: Utilizes self-supervised techniques to reduce the need for labeled data.
- **Controllable Feature Flow**: Allows for precise control over the interpolation process.

## Reference
1. Lei, “Self-supervised medical slice interpolation network using controllable feature flow”, Expert Systems with Applications (2024).
