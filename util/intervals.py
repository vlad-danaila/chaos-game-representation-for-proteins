import numpy as np
import portion as p

EPSILON = 1e-10

# args: numpy arrays
def boundaries(center_and_spread):
    center = center_and_spread[0]
    spread = center_and_spread[1]
    return p.closed(center - spread, center + spread)

# args: portion intervals
def span(interval: p.Interval):
    if interval == p.empty():
        return 0
    return interval.enclosure.upper - interval.enclosure.lower

# args: numpy arrays
def iou(interval_1, interval_2):
    boundaries_1 = boundaries(interval_1)
    boundaries_2 = boundaries(interval_2)
    intersection = boundaries_1.intersection(boundaries_2)
    union = (boundaries_1.union(boundaries_2)).enclosure
    return (span(intersection) + EPSILON) / (span(union) + EPSILON)

if __name__ == '__main__':
    interval_1 = np.array([2, 2])
    interval_2 = np.array([2, 0])
    print(iou(interval_1, interval_2))
