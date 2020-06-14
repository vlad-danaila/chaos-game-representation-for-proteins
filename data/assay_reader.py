from typing import List
import collections
import portion as p
import constants
from Bio import SeqIO
import os.path
import numpy as np

def skip_header(file):
    file.readline()

def read_virus_fasta_sequences(fasta_file_path):
    virus_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        virus_id = id_split[-2]
        seq = str(seq_record.seq)[:-1]
        virus_seq_dict[virus_id] = seq
    return virus_seq_dict

def read_antibody_fasta_sequences(fasta_file_path):
    antibody_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('_')
        antibody_id = id_split[0]
        seq = str(seq_record.seq)
        antibody_seq_dict[antibody_id] = seq
    return antibody_seq_dict

def curent_folder():
    return os.path.split(__file__)[0] + os.path.sep

VIRUS_SEQ_DICT = read_virus_fasta_sequences(curent_folder() + constants.VIRUS_SEQ)
ANTIBODY_HEAVY_SEQ_DICT = read_antibody_fasta_sequences(curent_folder() + constants.ANTIBODY_HEAVY_CHAIN_SEQ)
ANTIBODY_LIGHT_SEQ_DICT = read_antibody_fasta_sequences(curent_folder() + constants.ANTIBODY_LIGHT_CHAIN_SEQ)

class Assay():

    def __init__(self, antibody: str, virus: str, ic50: List[p.interval.Interval] = None, ic80: List[p.interval.Interval] = None):
        self.antibody = antibody
        self.virus = virus
        self.ic50 = ic50
        self.ic80 = ic80

    def __repr__(self):
        return '[antibody/virus=[{}/{}] ic50={} ic80={}]'.format(self.antibody, self.virus, self.ic50, self.ic80)

    def antibody_light_seq(self):
        return ANTIBODY_LIGHT_SEQ_DICT[self.antibody]

    def antibody_heavy_seq(self):
        return ANTIBODY_HEAVY_SEQ_DICT[self.antibody]

    def virus_seq(self):
        return VIRUS_SEQ_DICT[self.virus]

    def _interval_enclosure(self, measurements: List[p.Interval]):
        if measurements == None or len(measurements) == 0:
            return None
        union: p.interval.Interval = measurements[0]
        for i in range(1, len(measurements)):
            union = union | measurements[i]
        return union.enclosure

    def _ic50_interval_enclosure(self):
        return self._interval_enclosure(self.ic50)

    def _ic80_interval_enclosure(self):
        return self._interval_enclosure(self.ic80)

    def _interval_center_and_spread(self, interval_list: List[p.Interval]):
        interval = self._interval_enclosure(interval_list)
        if interval == None or interval == p.empty():
            return None
        low, high = interval.lower, min(interval.upper, constants.INTERVALS_UPPER_BOUND)
        center = (low + high) / 2
        spread = high - center
        return np.array([center, spread])

    def ic50_center_and_spread(self):
        return self._interval_center_and_spread(self.ic50)

    def ic80_center_and_spread(self):
        return self._interval_center_and_spread(self.ic80)

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
            antibody_id, virus_id = key
            assay = Assay(antibody_id, virus_id)
            if 'ic50' in assays_dict[key]:
                assay.ic50 = assays_dict[key]['ic50']
            if 'ic80' in assays_dict[key]:
                assay.ic80 = assays_dict[key]['ic80']
            assays.append(assay)

        return assays

if __name__ == '__main__':
    assay_reader = AssayReader(
        constants.ASSAY_FILE_PATH, constants.VIRUS_SEQ, constants.ANTIBODY_LIGHT_CHAIN_SEQ, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    assays = assay_reader.read_file()

    print(len(assays))

# Problematic example
# ('VRC34.01', '25710_2_43') {'ic50': [[50.0,+inf), [3.09], [0.128]], 'ic80': [[50.0,+inf), [100.0,+inf), [6.22]]})