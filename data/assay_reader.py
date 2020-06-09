from Bio import SeqIO
from typing import List
import collections

def skip_header(file):
    file.readline()

class Assay():

    def __init__(self, antibody_ids, virus_id, ic50):
        self.antibody_ids = antibody_ids
        self.virus_id = virus_id
        self.ic50 = ic50

    def __repr__(self):
        return '(antibodys = {} virus = {} ic50 = {})'.format(self.antibody_ids, self.virus_id, self.ic50)

class AssayReader():

    def __init__(self, assay_file_path):
        self.assays = []
        self.read_file(assay_file_path)

    def read_file(self, assay_file_path):
        with open(assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split()
                antibody_ids = self.find_antibody_ids(line_split[0])
                virus_id = line_split[1]
                ic50 = self.find_ic50_from_line_split(line_split)
                assay = Assay(antibody_ids, virus_id, ic50)
                self.assays.append(assay)

    def find_antibody_ids(self, antibodys_as_text: str):
        if '+' in antibodys_as_text:
            return antibodys_as_text.split('+')
        return [ antibodys_as_text ]

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

def print_antibodies_vs_viruses_stats(assays_list: List[Assay], virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict):
    counter_virus_vs_antibody = collections.defaultdict(lambda: 0)

    counter_virus_with_seq_assays = 0
    counter_virus_with_seq = collections.defaultdict(lambda: 0)

    counter_antibody_with_seq_light_assays = 0
    counter_antibody_with_seq_heavy_assays = 0
    counter_antibody_with_seq_both_assays = 0
    counter_antibody_with_seq_or_assays = 0
    counter_all_known = 0

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
        counter_all_known += (is_known_antibody_light and is_known_antibody_heavy and is_known_virus_seq)

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

def are_all_antibodies_known(antibody_ids, known_antibody_ids):
    for antibody_id in antibody_ids:
        if not antibody_id in known_antibody_ids:
            return False
    return True

if __name__ == '__main__':
    assay_reader = AssayReader('assay_CATNAP.txt')
    print(len(assay_reader.assays), 'lab records')

    virus_seq_dict = read_virus_fasta_sequences("virseqs_aa_CATNAP.fasta")
    print(len(virus_seq_dict), 'virus sequences')

    antibody_heavy_seq_dict = read_antibody_fasta_sequences("heavy_seqs_aa_CATNAP.fasta")
    print(len(antibody_heavy_seq_dict), 'antibody (heavy protein chain) sequences')

    antibody_light_seq_dict = read_antibody_fasta_sequences("light_seqs_aa_CATNAP.fasta")
    print(len(antibody_light_seq_dict), 'antibody (light protein chain) sequences')

    print_antibodies_vs_viruses_stats(assay_reader.assays, virus_seq_dict, antibody_heavy_seq_dict, antibody_light_seq_dict)