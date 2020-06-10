from data.assay_reader import Assay, FilteredAssayReader
import data.data_split
from os.path import join
import constants
from data.data_split import DatasetSplit, read_random_splits_from_file

DATA = 'data'



if __name__ == '__main__':
    assays = read_data()
    dataset_split = read_random_splits_from_file(join('..', DATA, constants.RANDOM_SPLIT))
