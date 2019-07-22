from .xml_style import XMLDataset
from .registry import DATASETS

@DATASETS.register_module
class DadiDetectionDataset(XMLDataset):

    CLASSES = ('text', 'title1', 'title2', 'title3', 'table', 'flag', 'twoline')

    def __init__(self, **kwargs):
        super(DadiDetectionDataset, self).__init__(**kwargs)
