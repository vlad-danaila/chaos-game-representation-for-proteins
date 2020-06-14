from typing import List
import collections
import constants
from data.assay_reader import read_antibody_fasta_sequences, read_virus_fasta_sequences
from scipy.stats import ttest_1samp, variation
import math
import numpy as np

P_VALUE_TRESHOLD = 0.003
SCALED_STD_TRESHOLD = 0.01

EXCLUDE_SINGLE_EXPERIMENTS = False
EXCLUDE_EXPERIMENTS_WIHT_HIGH_VARIANCE = True

class AssayMultipleAntibodies():

    def __init__(self, antibody_ids, virus_id, ic50):
        self.antibody_ids = antibody_ids
        self.virus_id = virus_id
        self.ic50 = ic50

    def __repr__(self):
        return '(antibodys = {} virus = {} ic50 = {})'.format(self.antibody_ids, self.virus_id, self.ic50)

def skip_header(file):
    file.readline()

class AssayReader():

    def __init__(self, assay_file_path):
        self.assay_file_path = assay_file_path

    def read_file(self):
        assays = []
        with open(self.assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split()
                # skip if there are multiple antibodies in the same assay
                if '+' in line_split[0]:
                    continue
                antibody_data = self.find_antibody_data(line_split[0])
                virus_id = line_split[1]
                ic50 = self.find_ic50_from_line_split(line_split)
                assay = Assay(antibody_data, virus_id, ic50)
                assays.append(assay)
        return assays

    def find_antibody_data(self, antibodys_as_text: str):
        if '+' in antibodys_as_text:
            raise Exception('Does not support multiple antibody ids.')
        return antibodys_as_text

    def find_ic50_from_line_split(self, line_split):
        ic50 = None
        for i in range(len(line_split) - 1, 0, -1):
            elem = line_split[i]
            try:
                # check if it's a PubMed id
                if int(elem) > 100_000:
                    break
            except ValueError:
                pass
            if elem.startswith('<') or elem.startswith('>'):
                elem = elem[1:]
            try:
                ic50 = float(elem)
            except ValueError:
                pass
        return ic50

class AssayMultipleAntibodyReader(AssayReader):

    def read_file(self):
        assays = []
        with open(self.assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split()
                antibody_data = self.find_antibody_data(line_split[0])
                virus_id = line_split[1]
                ic50 = self.find_ic50_from_line_split(line_split)
                assay = AssayMultipleAntibodies(antibody_data, virus_id, ic50)
                assays.append(assay)
        return assays

    def find_antibody_data(self, antibodys_as_text: str):
        if '+' in antibodys_as_text:
            return antibodys_as_text.split('+')
        return [ antibodys_as_text ]

class FilteredAssayReader(AssayReader):

    def __init__(self, assay_file_path, virus_seq_file_path, antibody_light_chain_file_path, antibody_heavy_chain_file_path):
        super().__init__(assay_file_path)
        self.virus_seq_file_path = virus_seq_file_path
        self.antibody_light_chain_file_path = antibody_light_chain_file_path
        self.antibody_heavy_chain_file_path = antibody_heavy_chain_file_path

    def read_file(self):
        assays_dict = collections.defaultdict(lambda: [])
        virus_seq_dict = read_virus_fasta_sequences(self.virus_seq_file_path)
        antibody_light_seq_dict = read_antibody_fasta_sequences(self.antibody_light_chain_file_path)
        antibody_heavy_seq_dict = read_antibody_fasta_sequences(self.antibody_heavy_chain_file_path)
        with open(self.assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split()
                # keep only fixed values, exclude open intervals
                if '>' in line or '<' in line:
                    continue
                # skip if there are multiple antibodies in the same assay
                if '+' in line_split[0]:
                    continue
                antibody_id = self.find_antibody_data(line_split[0])
                virus_id = line_split[1]
                ic50 = self.find_ic50_from_line_split(line_split)

                is_known_virus_seq = virus_id in virus_seq_dict
                is_known_antibody_light = antibody_id in antibody_light_seq_dict
                is_known_antibody_heavy = antibody_id in antibody_heavy_seq_dict

                if is_known_virus_seq and is_known_antibody_light and is_known_antibody_heavy:
                    assays_dict[(antibody_id, virus_id)].append(ic50)

        assays = []
        for (antibody_id, virus_id), ic50 in assays_dict.items():
            try:
                assay = self.aggregate_ic50(antibody_id, virus_id, ic50)
                assays.append(assay)
            except UnstableAssayException:
                continue
        return assays

    def aggregate_ic50(self, antibody_id, virus_id, ic50) -> Assay:
        # if all elements are equal
        if len(set(ic50)) == 1:
            if constants.EXCLUDE_SINGLE_EXPERIMENTS:
                raise UnstableAssayException()
            return Assay(antibody_id, virus_id, ic50[0])
        elif len(ic50) == 2:
            mean, scaled_std = self.scaled_std(ic50)
            if scaled_std > constants.SCALED_STD_TRESHOLD and constants.EXCLUDE_EXPERIMENTS_WIHT_HIGH_VARIANCE:
                raise UnstableAssayException()
            return Assay(antibody_id, virus_id, mean)
        elif len(ic50) > 2:
            included = self.filter_outliers_based_on_p_value(ic50)
            mean, scaled_std = self.scaled_std(included)
            if scaled_std > constants.SCALED_STD_TRESHOLD and constants.EXCLUDE_EXPERIMENTS_WIHT_HIGH_VARIANCE:
                raise UnstableAssayException()
            return Assay(antibody_id, virus_id, mean)

    def scaled_std(self, values):
        values = np.array(values)
        mean = values.mean()
        std = values.std()
        scaled_std = std / mean
        return mean, scaled_std

    def filter_outliers_based_on_p_value(self, ic50):
        included = []
        # Filter outliers using p value test
        for i in range(len(ic50)):
            ic50_excluded = list(ic50)
            del ic50_excluded[i]
            t_stat, p_value_double_tailed = ttest_1samp(ic50_excluded, ic50[i])
            if p_value_double_tailed > constants.P_VALUE_TRESHOLD:
                included.append(ic50[i])
        return included

class UnstableAssayException(Exception):
    pass

def print_multiple_antibodies_vs_viruses_stats(assays_list: List[AssayMultipleAntibodies], virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict):
    counter_virus_vs_antibody = collections.defaultdict(lambda: 0)

    counter_virus_with_seq_assays = 0
    counter_virus_with_seq = collections.defaultdict(lambda: 0)

    counter_antibody_with_seq_light_assays = 0
    counter_antibody_with_seq_heavy_assays = 0
    counter_antibody_with_seq_both_assays = 0
    counter_antibody_with_seq_or_assays = 0
    counter_all_known = 0
    counter_all_known_and_single_antibody = 0

    for assay in assays_list:
        counter_virus_vs_antibody[assay.virus_id] += 1
        is_known_virus_seq = assay.virus_id in virus_seq_dict
        if is_known_virus_seq:
            counter_virus_with_seq_assays += is_known_virus_seq
            counter_virus_with_seq[assay.virus_id] += is_known_virus_seq
        is_known_antibody_light = are_all_antibodies_known(assay.antibody_ids, antibody_light_seq_dict)
        is_known_antibody_heavy = are_all_antibodies_known(assay.antibody_ids, antibody_heavy_seq_dict)
        counter_antibody_with_seq_light_assays += is_known_antibody_light
        counter_antibody_with_seq_heavy_assays += is_known_antibody_heavy
        counter_antibody_with_seq_both_assays += (is_known_antibody_light and is_known_antibody_heavy)
        counter_antibody_with_seq_or_assays += (is_known_antibody_light or is_known_antibody_heavy)
        all_known = is_known_antibody_light and is_known_antibody_heavy and is_known_virus_seq
        counter_all_known += all_known
        counter_all_known_and_single_antibody += (all_known and len(assay.antibody_ids) == 1)

    print(len(counter_virus_vs_antibody), 'distinct viruses in assays')
    print('Max antibodies tests for virus', max(counter_virus_vs_antibody.values()))
    print('Min antibodies tests for virus', min(counter_virus_vs_antibody.values()))
    print('Assays with known virus seq', counter_virus_with_seq_assays)
    print('Viruses with known seq', len(counter_virus_with_seq))
    print('Assays with antibodies with known light seq', counter_antibody_with_seq_light_assays)
    print('Assays with antibodies with known heavy seq', counter_antibody_with_seq_heavy_assays)
    print('Assays with antibodies with known heavy & light seq', counter_antibody_with_seq_both_assays)
    print('Assays with antibodies with known heavy or light seq', counter_antibody_with_seq_or_assays)
    print('Assays with all known', counter_all_known)
    print('Assays with all known an single antibody', counter_all_known_and_single_antibody)
    # Display how many assays per virus stain
    # for key in counter_virus_vs_antibody:
    #     print(key, counter_virus_vs_antibody[key])

def are_all_antibodies_known(antibody_ids, known_antibody_ids):
    for antibody_id in antibody_ids:
        if not antibody_id in known_antibody_ids:
            return False
    return True

def print_single_antibodies_vs_viruses_stats(assays_list: List[AssayMultipleAntibodies], virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict):
    counter_virus_vs_antibody = collections.defaultdict(lambda: 0)

    counter_virus_with_seq_assays = 0
    counter_virus_with_seq = collections.defaultdict(lambda: 0)

    counter_antibody_with_seq_light_assays = 0
    counter_antibody_with_seq_heavy_assays = 0
    counter_antibody_with_seq_both_assays = 0
    counter_antibody_with_seq_or_assays = 0
    counter_all_known = 0
    counter_all_known_and_single_antibody = 0

    for assay in assays_list:
        counter_virus_vs_antibody[assay.virus_id] += 1
        is_known_virus_seq = assay.virus_id in virus_seq_dict
        if is_known_virus_seq:
            counter_virus_with_seq_assays += is_known_virus_seq
            counter_virus_with_seq[assay.virus_id] += is_known_virus_seq
        is_known_antibody_light = assay.antibody_id in antibody_light_seq_dict
        is_known_antibody_heavy = assay.antibody_id in antibody_heavy_seq_dict
        counter_antibody_with_seq_light_assays += is_known_antibody_light
        counter_antibody_with_seq_heavy_assays += is_known_antibody_heavy
        counter_antibody_with_seq_both_assays += (is_known_antibody_light and is_known_antibody_heavy)
        counter_antibody_with_seq_or_assays += (is_known_antibody_light or is_known_antibody_heavy)
        all_known = is_known_antibody_light and is_known_antibody_heavy and is_known_virus_seq
        counter_all_known += all_known

    print(len(counter_virus_vs_antibody), 'distinct viruses in assays')
    print('Max antibodies tests for virus', max(counter_virus_vs_antibody.values()))
    print('Min antibodies tests for virus', min(counter_virus_vs_antibody.values()))
    print('Assays with known virus seq', counter_virus_with_seq_assays)
    print('Viruses with known seq', len(counter_virus_with_seq))
    print('Assays with antibodies with known light seq', counter_antibody_with_seq_light_assays)
    print('Assays with antibodies with known heavy seq', counter_antibody_with_seq_heavy_assays)
    print('Assays with antibodies with known heavy & light seq', counter_antibody_with_seq_both_assays)
    print('Assays with antibodies with known heavy or light seq', counter_antibody_with_seq_or_assays)
    print('Assays with all known', counter_all_known)

if __name__ == '__main__':
    print('MULTIPLE ANTIBODIES VERSION -----------------')

    assay_multiple_antibodies_reader = AssayMultipleAntibodyReader(constants.ASSAY_FILE_PATH)
    assays = assay_multiple_antibodies_reader.read_file()
    print(len(assays), 'lab multiple antibodies records')

    virus_seq_dict = read_virus_fasta_sequences(constants.VIRUS_SEQ)
    print(len(virus_seq_dict), 'virus sequences')
    print('Length of one sequence', len(next(iter(virus_seq_dict.values()))))

    antibody_heavy_seq_dict = read_antibody_fasta_sequences(constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    print(len(antibody_heavy_seq_dict), 'antibody (heavy protein chain) sequences')

    antibody_light_seq_dict = read_antibody_fasta_sequences(constants.ANTIBODY_LIGHT_CHAIN_SEQ)
    print(len(antibody_light_seq_dict), 'antibody (light protein chain) sequences')

    print_multiple_antibodies_vs_viruses_stats(assays, virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict)

    print('SINGLE ANTIBODIES VERSION -----------------')

    assay_reader = AssayReader(constants.ASSAY_FILE_PATH)
    assays = assay_reader.read_file()
    print(len(assays), 'lab single antibodies records')

    print_single_antibodies_vs_viruses_stats(assays, virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict)

    print('FILTERED ASSAYS VERSION -----------------')

    assay_filtered_antibodies_reader = FilteredAssayReader(
        constants.ASSAY_FILE_PATH, constants.VIRUS_SEQ, constants.ANTIBODY_LIGHT_CHAIN_SEQ, constants.ANTIBODY_HEAVY_CHAIN_SEQ)
    assays = assay_filtered_antibodies_reader.read_file()
    print('Filtered assays', len(assays))