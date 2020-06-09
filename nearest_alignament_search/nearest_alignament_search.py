from data.assay_reader import Assay, FilteredAssayReader
import data.data_split
from os.path import join
import constants

def read_data():
    DATA = 'data'
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
    print('Filtered assays', len(assays))
    print(assays)