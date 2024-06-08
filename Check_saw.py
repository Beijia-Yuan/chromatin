#!/usr/bin/env python
# coding: utf-8

import numpy as np

def no_consecutive_repeated_pairs(arrays, thres):
    b = int(thres)+1
    for i in range(len(arrays)-1):  
        for j in range(i+b,len(arrays)-1): 
            if np.linalg.norm(arrays[i]-arrays[j]) < thres:
                return False
    return True


def continuous(points_list):
    matrix = np.stack(points_list)
    transposed_matrix = matrix.T
    s = np.diff(transposed_matrix).T
    norms = np.linalg.norm(s, axis=1)
    return np.all(np.isclose(norms, 1))


def check_SAW(arrays, thres):
    return (continuous(arrays) & no_consecutive_repeated_pairs(arrays, thres))


def check_duplicate(points_list):
    seen = set()
    dupes = []

    for y in points_list:
        x = tuple(y)
        if x in seen:
            dupes.append(x)
        else:
            seen.add(x)
    return(dupes)







