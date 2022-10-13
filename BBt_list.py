# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:39:18 2021

@author: masahiro
"""

### takes a list of bi-adjacency list and flag for
### whether computing B*B' (w.r.t pople) or B'B (w,r,t location)
### return a list of square matrices expressing the infection of
### nodes of the same type from a timestep to the next timestep 

def create_BBt_list(bi_adjacency_list, which="p"):

    BBt_list = []
    if which=="p":
        for i in range(len(bi_adjacency_list)-1):
            BBt_list.append(bi_adjacency_list[i]@bi_adjacency_list[i+1].transpose())
        BBt_list.append(bi_adjacency_list[-1]@bi_adjacency_list[0].transpose())
    else:
        for i in range(len(bi_adjacency_list)-1):
            BBt_list.append(bi_adjacency_list[i].transpose()@bi_adjacency_list[i+1])
        BBt_list.append(bi_adjacency_list[-1].transpose()@bi_adjacency_list[0])
    return BBt_list
