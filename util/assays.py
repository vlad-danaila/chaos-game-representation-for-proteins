from data.assay_reader import Assay
from typing import List
import numpy as np
import portion as p

def assays_intervals_mean(assays: List[Assay]):
    total = np.zeros(2)
    for assay in assays:
        total += assay.ic50_center_and_spread()
    return total / len(assays)

if __name__ == '__main__':
    a1 = Assay('', '', [p.closed(1, 2), p.closed(3, 5)])
    a2 = Assay('', '', [p.closed(10, 11), p.closed(10.5, 12)])
    print(assays_intervals_mean([a1, a2]))