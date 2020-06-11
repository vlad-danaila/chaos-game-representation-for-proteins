from data.assay_reader import FilteredAssayReader
import random
import constants
import numpy as np
import os.path
import matplotlib.pyplot as plt

def curent_folder():
    return os.path.split(__file__)[0] + os.path.sep

def read_data():
    assay_filtered_antibodies_reader = FilteredAssayReader(
        curent_folder() + constants.ASSAY_FILE_PATH,
        curent_folder() + constants.VIRUS_SEQ,
        curent_folder() + constants.ANTIBODY_LIGHT_CHAIN_SEQ,
        curent_folder() + constants.ANTIBODY_HEAVY_CHAIN_SEQ
    )
    assays = assay_filtered_antibodies_reader.read_file()
    return assays

if __name__ == '__main__':
    assays = read_data()
    ic50 = list(map(lambda assay: assay.ic50, assays))
    print(len(ic50))
    print(max(ic50))

    tresholds = np.linspace(0, 55, 1000)
    counts = []

    for i in range(len(tresholds)):
        counter = 0
        for val in ic50:
            counter += val < tresholds[i]
        counts.append(counter)
        print(i)

    plt.plot(tresholds, counts)
    plt.show()