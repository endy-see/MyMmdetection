from .xml_style import XMLDataset


class HandeTableTitleWithOtherTextDataset(XMLDataset):

    CLASSES = ('balancesheet', 'cashflowstatement', 'incomestatement', 'othertext')

    def __init__(self, **kwargs):
        super(HandeTableTitleWithOtherTextDataset, self).__init__(**kwargs)
