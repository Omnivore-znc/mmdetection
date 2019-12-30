
from .registry import DATASETS
from .txt_style import TXTDataset
from .json_style import JSONDataset


@DATASETS.register_module
# class BodyKeypointDataset(TXTDataset):
class BodyKeypointDataset(JSONDataset):
    CLASSES = ('labeled-unvisible', 'labeled-visible')
    def __init__(self, min_size=None, **kwargs):
        super(BodyKeypointDataset, self).__init__(**kwargs)
