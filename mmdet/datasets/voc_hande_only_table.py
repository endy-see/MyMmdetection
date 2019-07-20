from .xml_style import XMLDataset
from .registry import DATASETS

@DATASETS.register_module
class HandeOnlyTableDataset(XMLDataset):

    CLASSES = ('toptable', 'bottomtable', 'topbottomtable', 'othertable')

    def __init__(self, **kwargs):
        super(HandeOnlyTableDataset, self).__init__(**kwargs)
