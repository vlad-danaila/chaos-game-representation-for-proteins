from data.data_split import read_data_by_serialized_random_split
from data.assay_reader import Assay
from typing import List
import numpy as np
import constants
from util.timer import timer_start, timer_end
from util.intervals import iou

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

def k_neibhours(assays: List[Assay], compared_assay: Assay, k_neibhours: int):
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

    total_weighted_intervals, total_weights = np.zeros(2), np.zeros(2)

    for i in range(k_neibhours):
        sort_index = sort_indexes[i]
        dist = distances[sort_index]
        dist_weight = 1 / dist
        assay = assays[sort_index]
        interval = assay.ic50_center_and_spread()
        total_weighted_intervals += dist_weight * interval
        total_weights += dist_weight

    return total_weighted_intervals / total_weights

if __name__ == '__main__':
    train_assays, val_assays, test_assays = read_data_by_serialized_random_split()
    abs_err_total = np.zeros(2)
    squared_err_total = np.zeros(2)
    iou_total = 0
    for i in range(len(test_assays)):
        test_assay = test_assays[i]
        nebhour_interval = k_neibhours(train_assays, test_assay, constants.K_NEIBHOURS)
        expected = test_assay.ic50_center_and_spread()
        abs_err = np.abs(nebhour_interval - expected)
        abs_err_total += abs_err
        squared_err_total += (abs_err ** 2)
        iou_total += iou(nebhour_interval, expected)
        print('Processed', i / len(test_assays), '%')
        if i == 2:
            break
    abs_err_mean = abs_err_total / len(test_assays)
    squared_err_mean = squared_err_total / len(test_assays)
    iou_mean = iou_total / len(test_assays)

    print('fianl', abs_err_mean, squared_err_mean, iou_mean)
    print('len', len(test_assays))



# TODO compute and R2
# TODO compute simultaneously for k = 1, 3, 5, 10, 30, 50, 100, 300, 500, 1000
# TODO time it again
# TODO checkpointing
