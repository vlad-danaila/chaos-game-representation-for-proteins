from typing import Tuple, List
import collections
import constants
from data.entity import read_antibody_fasta_sequences, read_virus_fasta_sequences
from scipy.stats import ttest_1samp, variation
import math
import numpy as np
import portion as p

def skip_header(file):
    file.readline()

class Assay():

    def __init__(self, antibody: str, virus: str, ic50: List[p.interval.Interval], ic80: List[p.interval.Interval]):
        self.virus = virus
        self.antibody = antibody
        self.ic50 = ic50
        self.ic80 = ic80

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

    def read_interval(self, value: str):
        try:
            if value.startswith('>'):
                return p.closedopen(float(value[1:]), p.inf)
            elif value.startswith('<'):
                return p.closed(0, float(value[1:]))
            else:
                return p.singleton(float(value))
        except ValueError:
            return p.empty()

    def read_file(self):
        # A dict of dicts of lists, first level is virus/antibody pair,
        # second level is ic50, ic80, and the last level are lists of experiments values
        assays_dict = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [])
        )

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
                ic50 = self.read_interval(line_split[-3])
                ic80 = self.read_interval(line_split[-2])
                # We don't use the id50
                # id50 = self.read_interval(line_split[-1])

                is_known_virus_seq = virus_id in virus_seq_dict
                is_known_antibody_light = antibody_id in antibody_light_seq_dict
                is_known_antibody_heavy = antibody_id in antibody_heavy_seq_dict

                if is_known_virus_seq and is_known_antibody_light and is_known_antibody_heavy:
                    if ic50 != p.empty():
                        assays_dict[(antibody_id, virus_id)]['ic50'].append(ic50)
                    if ic80 != p.empty():
                        assays_dict[(antibody_id, virus_id)]['ic80'].append(ic80)

        assays = []
        for key in assays_dict:
            print(key, assays_dict[key])
            # antibody_id, virus_id = key

        #     print(antibody_id, virus_id, assays_dict[key])


class UnstableAssayException(Exception):
    pass

if __name__ == '__main__':
    assay_reader = AssayReader(
        constants.ASSAY_FILE_PATH, constants.VIRUS_SEQ, constants.ANTIBODY_LIGHT_CHAIN_SEQ, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    assays = assay_reader.read_file()