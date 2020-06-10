from data.assay_reader import Assay, FilteredAssayReader
import random
import constants
import numpy as np

class DatasetSplit():
    def __init__(self, train: list, val: list, test: list):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self):
        return 'train: {} val: {} test: {}'.format(len(self.train), len(self.val), len(self.test))

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

def read_data():
    assay_filtered_antibodies_reader = FilteredAssayReader(
        constants.ASSAY_FILE_PATH, constants.VIRUS_SEQ, constants.ANTIBODY_LIGHT_CHAIN_SEQ, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    assays = assay_filtered_antibodies_reader.read_file()
    return assays

def read_data_by_split(dataset_split):
    assays = np.array(read_data())
    train_assays = assays[dataset_split.train]
    val_assays = assays[dataset_split.val]
    test_assays = assays[dataset_split.test]
    return train_assays, val_assays, test_assays

if __name__ == '__main__':
    # dataset_split = get_random_splits(82988, 0.8, 0.1, 0.1)
    # dataset_split.serilize(constants.RANDOM_SPLIT)

    dataset_split = read_random_splits_from_file(constants.RANDOM_SPLIT)

    train_assays, val_assays, test_assays = read_data_by_split(dataset_split)
