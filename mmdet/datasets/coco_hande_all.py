from .coco import CocoDataset

class HandeTable(CocoDataset):
    CLASSES = ('balancesheet', 'cashflowstatement', 'incomestatement', 'othertext', 'toptable', 'bottomtable', 'topbottomtable', 'othertable')
