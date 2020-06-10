from torch.utils.data import Dataset
from data.assay_reader import FilteredAssayReader
import constants

class AssayDataset(Dataset):

    def __init__(self, assays):
        self.assays = assays

    # TODO new class for x and y
    # TODO How to better handle small y
    def __getitem__(self, i):
        self.assays[i]

    def __len__(self):
        return len(self.assays)