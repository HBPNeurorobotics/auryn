from online_median_heap import OnlineMedianFinder
import numpy as np
import ipdb
import heapq

try:
    omf = OnlineMedianFinder()
    for i in range(20):
        omf.add_element(i)
        print(omf.currentMedian)

    check = np.array(omf.bigElements.getheap() + omf.smallElements.getheap())
    print(check)
    print(np.median(check))
    print(omf.smallElements)
    print(omf.bigElements)
    for i in range(10):
        omf.remove_element(i)
        print(omf.currentMedian)
except ValueError:
    print('ERROR at', i)
    print(omf.smallElements.getheap())
    print(omf.bigElements.getheap())
    print(omf.currentMedian)
