import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep_ours
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class TMB(nn.Module):
    def __init__(self):
        super(TMB, self).__init__()
        self.t_process = nn.Sequential(*[
            nn.Conv2d( 1, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.Sigmoid()    
        ])
        self.f_process = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ])
        self.kernel_size=3
        self.kernel = nn.Sequential(*[
            nn.Conv2d( 1, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64 * self.kernel_size * self.kernel_size, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ])
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
    #SCM
    def forward(self, x, t):
        t_orig = t
        b, c, h, w = x.size()
        feature = self.f_process(x)
        # channel modulation
        modulation_vector = self.t_process(t_orig)
        # spatial modulation
        kernel = self.kernel(t_orig).view(-1, 1, self.kernel_size, self.kernel_size)
        out = F.conv2d(feature.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2).view(b, -1, h, w)
        output = out * modulation_vector + self.conv33(x)
        return output

   

class DCN_Align(nn.Module):
    def __init__(self, nf=64, groups=8, use_time=True):
        super(DCN_Align, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1    
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.offset_conv5_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        self.offset_conv8_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv4_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.dcnpack = DCN_sep_ours(nf, nf, 5, stride=1, padding=2, dilation=1,
                            deformable_groups=4)
        self.fusion = nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.TMB = TMB()

        self.mask_net = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 100, 3, 1, 1, bias=True),
            nn.Sigmoid()])
        self.offset_x_net = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 100, 3, 1, 1, bias=True)])
        self.offset_y_net = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 100, 3, 1, 1, bias=True)])


    def forward(self, fea1, fea2, t=None, t_back=None):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        t_N = t.shape[1]
        offset = torch.cat([fea1, fea2], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        offset2 = self.lrelu(self.offset_conv5_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        offset3 = self.lrelu(self.offset_conv8_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset1 = self.offset_conv4_2(offset)

        offset = torch.cat([fea2, fea1], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        offset2 = self.lrelu(self.offset_conv5_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        offset3 = self.lrelu(self.offset_conv8_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset2 = self.offset_conv4_2(offset)
 
        intered_fea = []
        for i in range(t_N):
            offset_t_1 = self.TMB(base_offset1, t[:, i, :, :, :])
            mask_1 = self.mask_net(offset_t_1)
            offset_x_1 = self.offset_x_net(offset_t_1)
            offset_y_1 = self.offset_y_net(offset_t_1)
            aligned_fea2 = self.dcnpack(fea1, torch.cat((offset_x_1, offset_y_1), 1), mask_1)

            offset_t_2 = self.TMB(base_offset2, t_back[:, i, :, :, :])
            mask_2 = self.mask_net(offset_t_2)
            offset_x_2 = self.offset_x_net(offset_t_2)
            offset_y_2 = self.offset_y_net(offset_t_2)
            aligned_fea1 = self.dcnpack(fea2, torch.cat((offset_x_2, offset_y_2), 1), mask_2)

            aligned_fea = self.fusion(torch.cat((aligned_fea1, aligned_fea2), 1))
            intered_fea.append(aligned_fea)
        
        return intered_fea


class CFFNet(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, opt=None):
        super(CFFNet, self).__init__()
        self.opt = opt
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        input_channel = 1
    
        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(input_channel, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)

        self.dcn_align = DCN_Align(nf=nf, groups=groups, use_time=True)

        self.HRconv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(64, input_channel, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, t=None):   
        B, N, C, H, W = x.size()  # N input video frames
        
        t_B, t_N = t.shape[0], t.shape[1]
        t = t.view(t_B * t_N).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_back = 1 - t
        t = t.view(t_B, t_N, 1, 1, 1)
        t_back = t_back.view(t_B, t_N, 1, 1, 1)

        #### extract LR features
        L1_fea = self.conv_first(x.view(-1, C, H, W))
        L1_fea = self.feature_extraction(L1_fea) + L1_fea
        
        L1_fea = L1_fea.view(B, N, -1, H, W)
        fea1 = L1_fea[:, 0, :, :, :].clone()
        fea2 = L1_fea[:, 1, :, :, :].clone()
        # interpolate slice features
        to_slice_fea = self.dcn_align(fea1, fea2, t=t, t_back=t_back)

        stage1_fea = torch.stack(to_slice_fea, 1)
        B, T, C, H, W = stage1_fea.size()
        stage1_fea = stage1_fea.view(B*T, C, H, W) 

        # rec
        out1 = self.conv_last1(self.lrelu(self.HRconv1(stage1_fea)))
        _, _, K, G = out1.size()
        outs1 = out1.view(B, T, -1, K, G)

        return outs1