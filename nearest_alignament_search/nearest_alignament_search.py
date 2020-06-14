from data.data_split import read_data_by_serialized_random_split
from data.assay_reader import Assay
from typing import List
import numpy as np

def assay_distance(assay_1: Assay, assay_2: Assay):
    virus_seq_1, virus_seq_2 = assay_1.virus_seq(), assay_2.virus_seq()
    antibody_light_1, antibody_light_2 = assay_1.antibody_light_seq(), assay_2.antibody_light_seq()
    antibody_heavy_1, antibody_heavy_2 = assay_1.antibody_heavy_seq(), assay_2.antibody_heavy_seq()
    virus_seq_diff, antibody_lignt_seq_diff, antibody_heavy_seq_diff = 0, 0, 0

    assert len(virus_seq_1) == len(virus_seq_2)
    assert len(antibody_light_1) == len(antibody_light_2)
    assert len(antibody_heavy_1) == len(antibody_heavy_2)

    for i in range(len(virus_seq_1)):
        virus_seq_diff += virus_seq_1[i] != virus_seq_2[i]

    for i in range(len(antibody_light_1)):
        antibody_lignt_seq_diff += antibody_light_1[i] != antibody_light_2[i]

    for i in range(len(antibody_heavy_1)):
        antibody_heavy_seq_diff += antibody_heavy_1[i] != antibody_heavy_2[i]

    return virus_seq_diff, antibody_lignt_seq_diff, antibody_heavy_seq_diff

def compute_distances(assays: List[Assay], compared_assay: Assay):
    total_virus_diff, total_antoibody_light_diff, total_antoibody_heavy_diff = 0, 0, 0
    virus_diffs, antib_light_diffs, antib_heavy_diffs = [], [], []

    for assay in assays:
        virus_seq_diff, antibody_lignt_seq_diff, antibody_heavy_seq_diff = assay_distance(assay, compared_assay)
        total_virus_diff += virus_seq_diff
        total_antoibody_light_diff += antibody_lignt_seq_diff
        total_antoibody_heavy_diff += antibody_heavy_seq_diff
        virus_diffs.append(virus_seq_diff)
        antib_light_diffs.append(antibody_lignt_seq_diff)
        antib_heavy_diffs.append(antibody_heavy_seq_diff)

    mean_virus_diff = total_virus_diff / len(assays)
    mean_antibody_light_diff = total_antoibody_light_diff / len(assays)
    mean_antibody_heavy_diff = total_antoibody_heavy_diff / len(assays)

    virus_diffs = np.array(virus_diffs).reshape(-1, 1)
    antib_light_diffs = np.array(antib_light_diffs).reshape(-1, 1)
    antib_heavy_diffs = np.array(antib_heavy_diffs).reshape(-1, 1)

    virus_diffs = virus_diffs / mean_virus_diff
    antib_light_diffs = antib_light_diffs / mean_antibody_light_diff
    antib_heavy_diffs = antib_heavy_diffs / mean_antibody_heavy_diff

    distance_points = np.hstack([virus_diffs, antib_light_diffs, antib_heavy_diffs])
    distances = np.linalg.norm(distance_points, axis=1)

    sort_indexes = np.argsort(distances)

    for i in range(20):
        print(distances[sort_indexes[i]])
        print(assays[sort_indexes[i]])

if __name__ == '__main__':
    train_assays, val_assays, test_assays = read_data_by_serialized_random_split()
    counter = 1
    for test_assay in test_assays:
        compute_distances(train_assays, test_assay)
        break
        print('Processed', counter / len(test_assays), '%')
        counter += 1