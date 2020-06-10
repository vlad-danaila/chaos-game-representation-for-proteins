import constants
from Bio import SeqIO
import os.path

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

class AssayMultipleAntibodies():

    def __init__(self, antibody_ids, virus_id, ic50):
        self.antibody_ids = antibody_ids
        self.virus_id = virus_id
        self.ic50 = ic50

    def __repr__(self):
        return '(antibodys = {} virus = {} ic50 = {})'.format(self.antibody_ids, self.virus_id, self.ic50)

class Assay():

    def __init__(self, antibody_id, virus_id, ic50):
        self.antibody_id = antibody_id
        self.virus_id = virus_id
        self.ic50 = ic50

    def __repr__(self):
        return '(antibody = {} virus = {} ic50 = {})'.format(self.antibody_id, self.virus_id, self.ic50)

    def virus_seq(self):
        return VIRUS_SEQ_DICT[self.virus_id]

    def antibody_light_seq(self):
        return ANTIBODY_LIGHT_SEQ_DICT[self.antibody_id]

    def antibody_heavy_seq(self):
        return ANTIBODY_HEAVY_SEQ_DICT[self.antibody_id]