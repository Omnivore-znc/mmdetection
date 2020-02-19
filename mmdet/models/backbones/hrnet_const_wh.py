from .hrnet import HRNet
from ..registry import BACKBONES
import torch
import torch.nn as nn
import torch.nn.functional as F


@BACKBONES.register_module
class HRNetWH(HRNet):
    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 input_width=64,
                 input_height=128
                 ):
        super(HRNetWH, self).__init__(extra,
                 in_channels,
                 conv_cfg,
                 norm_cfg,
                 norm_eval,
                 with_cp,
                 zero_init_residual)
        # if len(out_indices)!=1:
        #     raise "Only support one out-indice in this class"
        self.input_height = input_height
        self.input_width = input_width

        # self.last_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=last_inp_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0),
        #     BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=config.DATASET.NUM_CLASSES,
        #         kernel_size=extra.FINAL_CONV_KERNEL,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        # )


    def forward(self, x):
        '''
        github： HRNet-Image-Classification
        :param x:
        :return:
        '''
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # # Classification Head
        # y = self.incre_modules[0](y_list[0])
        # for i in range(len(self.downsamp_modules)):
        #     y = self.incre_modules[i + 1](y_list[i + 1]) + \
        #         self.downsamp_modules[i](y)
        #
        # y = self.final_layer(y)

        # # Upsampling
        # x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        # x1 = F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear')
        # x2 = F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear')
        # x3 = F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')
        #
        # y = torch.cat([y_list[0], x1, x2, x3], 1)

        # x = self.last_layer(x)

        return y_list[0]
        # return y

@BACKBONES.register_module
class HRNetWH_V2(HRNet):
    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 input_width=64,
                 input_height=128
                 ):
        super(HRNetWH_V2, self).__init__(extra,
                 in_channels,
                 conv_cfg,
                 norm_cfg,
                 norm_eval,
                 with_cp,
                 zero_init_residual)
        # if len(out_indices)!=1:
        #     raise "Only support one out-indice in this class"
        self.input_height = input_height
        self.input_width = input_width

        # self.last_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=last_inp_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0),
        #     BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=config.DATASET.NUM_CLASSES,
        #         kernel_size=extra.FINAL_CONV_KERNEL,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        # )


    def forward(self, x):
        '''
        github： HRNet-Image-Classification
        :param x:
        :return:
        '''
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # # Classification Head
        # y = self.incre_modules[0](y_list[0])
        # for i in range(len(self.downsamp_modules)):
        #     y = self.incre_modules[i + 1](y_list[i + 1]) + \
        #         self.downsamp_modules[i](y)
        #
        # y = self.final_layer(y)

        # Upsampling
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        # x1 = F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear')
        # x3 = F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')

        # y = torch.cat([y_list[0], x1, x2, x3], 1)
        y = torch.cat([y_list[0], x2], 1)

        # x = self.last_layer(x)

        # return y_list[0]
        return y

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.norm1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     for i, layer_name in enumerate(self.res_layers):
    #         res_layer = getattr(self, layer_name)
    #         x = res_layer(x)
    #         if i in self.out_indices:
    #             return x
    #     raise ("No out-indice found in ResnetWH")
