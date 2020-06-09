from data.assay_reader import Assay, FilteredAssayReader

def cross_validate():
    pass

def train_eval():
    pass

class DatasetSplit():
    def __init__(self, train: list, val: list, test: list):
        self.train = train
        self.val = val
        self.test = test

def get_random_splits(dataset_size: int, train: float, validation: float, test: float):
    indexes = range(dataset_size)

if __name__ == '__main__':
    ASSAY_FILE_PATH = 'assay_CATNAP.txt'
    VIRUS_SEQ = "virseqs_aa_CATNAP.fasta"
    ANTIBODY_LIGHT_CHAIN_SEQ = "light_seqs_aa_CATNAP.fasta"
    ANTIBODY_HEAVY_CHAIN_SEQ = "heavy_seqs_aa_CATNAP.fasta"

    # assay_filtered_antibodies_reader = FilteredAssayReader(ASSAY_FILE_PATH, VIRUS_SEQ, ANTIBODY_LIGHT_CHAIN_SEQ, ANTIBODY_HEAVY_CHAIN_SEQ)
    # assays = assay_filtered_antibodies_reader.read_file()
    # print('Filtered assays', len(assays))

    get_random_splits(82988, 0.8, 0.1, 0.1)