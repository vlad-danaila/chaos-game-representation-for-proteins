from data.assay_reader import Assay, FilteredAssayReader
import data.data_split
from os.path import join
import constants
from data.data_split import DatasetSplit, read_random_splits_from_file

DATA = 'data'

def read_data():
    assay_filtered_antibodies_reader = FilteredAssayReader(
        join('..', DATA, constants.ASSAY_FILE_PATH),
        join('..', DATA, constants.VIRUS_SEQ),
        join('..', DATA, constants.ANTIBODY_LIGHT_CHAIN_SEQ),
        join('..', DATA, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    )
    assays = assay_filtered_antibodies_reader.read_file()
    return assays

if __name__ == '__main__':
    assays = read_data()
    dataset_split = read_random_splits_from_file(join('..', DATA, constants.RANDOM_SPLIT))
