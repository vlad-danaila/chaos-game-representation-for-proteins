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

    def serilize(self, file_path):
        with open(file_path, 'w') as file:
            file.write('train\n')
            file.write(' '.join([str(n) for n in self.train]) + '\n')
            file.write('validation\n')
            file.write(' '.join([str(n) for n in self.val]) + '\n')
            file.write('test\n')
            file.write(' '.join([str(n) for n in self.test]) + '\n')

def get_random_splits(dataset_size: int, train: float, validation: float, test: float) -> DatasetSplit:
    len_train = int(dataset_size * train)
    len_val = int(dataset_size * validation)
    len_test = int(dataset_size * test)
    print(len_train, 'records for training')
    print(len_val, 'records for validation')
    print(len_test, 'records for testing')
    indexes = set(range(dataset_size))
    validation_indexes = list(random.sample(indexes, len_val))
    indexes = indexes - set(validation_indexes)
    test_indexes = list(random.sample(indexes, len_test))
    indexes = list(indexes - set(test_indexes))
    # Shuffle all sequences
    random.shuffle(indexes)
    random.shuffle(validation_indexes)
    random.shuffle(test_indexes)
    return DatasetSplit(indexes, validation_indexes, test_indexes)

def read_random_splits_from_file(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        assert line == 'train\n'
        train_indexes = file.readline()
        train_indexes = [int(s) for s in train_indexes.split(' ')]
        line = file.readline()
        assert line == 'validation\n'
        val_indexes = file.readline()
        val_indexes = [int(s) for s in val_indexes.split(' ')]
        line = file.readline()
        assert line == 'test\n'
        test_indexes = file.readline()
        test_indexes = [int(s) for s in test_indexes.split(' ')]
        return DatasetSplit(train_indexes, val_indexes, test_indexes)

if __name__ == '__main__':
    ASSAY_FILE_PATH = 'assay_CATNAP.txt'
    VIRUS_SEQ = "virseqs_aa_CATNAP.fasta"
    ANTIBODY_LIGHT_CHAIN_SEQ = "light_seqs_aa_CATNAP.fasta"
    ANTIBODY_HEAVY_CHAIN_SEQ = "heavy_seqs_aa_CATNAP.fasta"
    RANDOM_SPLIT = 'random_split_indexes'

    # assay_filtered_antibodies_reader = FilteredAssayReader(ASSAY_FILE_PATH, VIRUS_SEQ, ANTIBODY_LIGHT_CHAIN_SEQ, ANTIBODY_HEAVY_CHAIN_SEQ)
    # assays = assay_filtered_antibodies_reader.read_file()
    # print('Filtered assays', len(assays))

    # dataset_split = get_random_splits(82988, 0.8, 0.1, 0.1)
    # dataset_split.serilize(RANDOM_SPLIT)

    dataset_split = read_random_splits_from_file(RANDOM_SPLIT)