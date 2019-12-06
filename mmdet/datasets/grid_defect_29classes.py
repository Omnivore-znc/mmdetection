from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDatasetGridDefect(XMLDataset):

    CLASSES = ('bj_bpmh', 'bj_bpps','bj_wkps','jyz_lw','jyz_pl',
               'sly_bjbmyw','sly_dmyw','jsxs','hxq_gjtps','xmbhyc',
               'yw_gkxfw','yw_nc','mcqdmsh','gbps','gjptwss',
               'bmwh','yxcr','wcaqm','wcgz','xy',
               'bjdsyc','ywzt_yfyc','hxq_gjbs','kgg_ybh','kgg_ybf',
               'gzzc', 'aqmzc', 'hxq_gjzc', 'xmbhzc')

    def __init__(self, **kwargs):
        super(VOCDatasetGridDefect, self).__init__(**kwargs)
        # if 'VOC2007' in self.img_prefix:
        #     self.year = 2007
        # elif 'VOC2012' in self.img_prefix:
        #     self.year = 2012
        # else:
        #     raise ValueError('Cannot infer dataset year from img_prefix')
        self.year = 201908
