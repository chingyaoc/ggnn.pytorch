from torch.utils.data import DataLoader

class bAbIDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(bAbIDataloader, self).__init__(*args, **kwargs)
