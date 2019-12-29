from .resnet import ResNet
from ..registry import BACKBONES

@BACKBONES.register_module
class ResNetWH(ResNet):
    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True,
                 input_width=64,
                 input_height=128):
        super(ResNetWH, self).__init__(depth,
                                       in_channels,
                                       num_stages,
                                       strides,
                                       dilations,
                                       out_indices,
                                       style,
                                       frozen_stages,
                                       conv_cfg,
                                       norm_cfg,
                                       norm_eval,
                                       dcn,
                                       stage_with_dcn,
                                       gcb,
                                       stage_with_gcb,
                                       gen_attention,
                                       stage_with_gen_attention,
                                       with_cp,
                                       zero_init_residual)
        if len(out_indices)!=1:
            raise "Only support one out-indice in this class"
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                return x
        raise "No out-indice found in ResnetWH"