from typing import Tuple, List
import collections
import constants
from data.entity import read_antibody_fasta_sequences, read_virus_fasta_sequences
from scipy.stats import ttest_1samp, variation
import math
import numpy as np

def skip_header(file):
    file.readline()

class Assay():

    def __init__(self, antibody: str, virus: str, ic50: Tuple[int, int], ic80: Tuple[int, int], id50: Tuple[int, int]):
        self.virus = virus
        self.antibody = antibody
        self.ic50 = ic50
        self.ic80 = ic80
        self.id50 = id50

class AssayReader():

    def __init__(self, assay_file_path, virus_seq_file_path, antibody_light_chain_file_path, antibody_heavy_chain_file_path):
        self.assay_file_path = assay_file_path
        self.virus_seq_file_path = virus_seq_file_path
        self.antibody_light_chain_file_path = antibody_light_chain_file_path
        self.antibody_heavy_chain_file_path = antibody_heavy_chain_file_path

    def find_antibody_data(self, antibodys_as_text: str):
        if '+' in antibodys_as_text:
            raise Exception('Does not support multiple antibody ids.')
        return antibodys_as_text

    # TODO
    def read_interval(self, value: str):
        pass

    def read_file(self):
        assays_dict = collections.defaultdict(lambda: [])

        virus_seq_dict = read_virus_fasta_sequences(self.virus_seq_file_path)
        antibody_light_seq_dict = read_antibody_fasta_sequences(self.antibody_light_chain_file_path)
        antibody_heavy_seq_dict = read_antibody_fasta_sequences(self.antibody_heavy_chain_file_path)

        with open(self.assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split('\t')
                assert len(line_split) == 7
                # skip if there are multiple antibodies in the same assay
                if '+' in line_split[0]:
                    continue
                antibody_id = self.find_antibody_data(line_split[0])
                virus_id = line_split[1]
                ic50 = line_split[-3]
                ic80 = line_split[-2]
                id50 = line_split[-1]
                print(ic50, ic80, id50, line_split)



                # ic50 = self.find_ic50_ic80_id50_from_line_split(line)


                # is_known_virus_seq = virus_id in virus_seq_dict
                # is_known_antibody_light = antibody_id in antibody_light_seq_dict
                # is_known_antibody_heavy = antibody_id in antibody_heavy_seq_dict
                #
                # if is_known_virus_seq and is_known_antibody_light and is_known_antibody_heavy:
                #     assays_dict[(antibody_id, virus_id)].append(ic50)

        # assays = []
        # for (antibody_id, virus_id), ic50 in assays_dict.items():
        #     try:
        #         assay = self.aggregate_ic50(antibody_id, virus_id, ic50)
        #         assays.append(assay)
        #     except UnstableAssayException:
        #         continue
        # return assays



class UnstableAssayException(Exception):
    pass

if __name__ == '__main__':
    assay_reader = AssayReader(
        constants.ASSAY_FILE_PATH, constants.VIRUS_SEQ, constants.ANTIBODY_LIGHT_CHAIN_SEQ, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    assays = assay_reader.read_file()
