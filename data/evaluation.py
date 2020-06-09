from data.assay_reader import Assay, FilteredAssayReader
import random

def cross_validate():
    pass

def train_eval():
    pass

class DatasetSplit():
    def __init__(self, train: list, val: list, test: list):
        self.train = train
        self.val = val
        self.test = test

def get_random_splits(dataset_size: int, train: float, validation: float, test: float) -> DatasetSplit:
    len_train = int(dataset_size * train)
    len_val = int(dataset_size * validation)
    len_test = int(dataset_size * test)
    print(len_train, 'records for training')
    print(len_val, 'records for validation')
    print(len_test, 'records for testing')
    indexes = set(range(dataset_size))
    validation_indexes = random.sample(indexes, len_val)
    indexes = indexes - set(validation_indexes)
    test_indexes = random.sample(indexes, len_test)
    indexes = indexes - set(test_indexes)
    return DatasetSplit(indexes, validation_indexes, test_indexes)

if __name__ == '__main__':
    ASSAY_FILE_PATH = 'assay_CATNAP.txt'
    VIRUS_SEQ = "virseqs_aa_CATNAP.fasta"
    ANTIBODY_LIGHT_CHAIN_SEQ = "light_seqs_aa_CATNAP.fasta"
    ANTIBODY_HEAVY_CHAIN_SEQ = "heavy_seqs_aa_CATNAP.fasta"

    # assay_filtered_antibodies_reader = FilteredAssayReader(ASSAY_FILE_PATH, VIRUS_SEQ, ANTIBODY_LIGHT_CHAIN_SEQ, ANTIBODY_HEAVY_CHAIN_SEQ)
    # assays = assay_filtered_antibodies_reader.read_file()
    # print('Filtered assays', len(assays))

    dataset_split = get_random_splits(82988, 0.8, 0.1, 0.1)
