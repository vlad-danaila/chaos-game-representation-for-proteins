from Bio import SeqIO

def skip_header(file):
    file.readline()

class Assay():

    def __init__(self, antibody_id, virus_id, ic50):
        self.antibody_id = antibody_id
        self.virus_id = virus_id
        self.ic50 = ic50

    def __repr__(self):
        return '(antibody = {} virus = {} ic50 = {})'.format(self.antibody_id, self.virus_id, self.ic50)

class AssayReader():

    def __init__(self, assay_file_path):
        self.assays = []
        self.read_file(assay_file_path)

    def read_file(self, assay_file_path):
        with open(assay_file_path, 'r') as file:
            skip_header(file)
            for line in file:
                line_split = line.split()
                antibody_id = line_split[0]
                virus_id = line_split[1]
                ic50 = self.find_ic50_from_line_split(line_split)
                assay = Assay(antibody_id, virus_id, ic50)
                self.assays.append(assay)

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

if __name__ == '__main__':
    assay_reader = AssayReader('assay_CATNAP.txt')
    print(len(assay_reader.assays), 'lab records')

    virus_seq_dict = read_virus_fasta_sequences("virseqs_aa_CATNAP.fasta")
    print(len(virus_seq_dict), 'virus sequences')

    antibody_heavy_seq_dict = read_antibody_fasta_sequences("heavy_seqs_aa_CATNAP.fasta")
    print(len(antibody_heavy_seq_dict), 'antibody (heavy protein chain) sequences')

    antibody_light_seq_dict = read_antibody_fasta_sequences("light_seqs_aa_CATNAP.fasta")
    print(len(antibody_light_seq_dict), 'virus (light protein chain) sequences')