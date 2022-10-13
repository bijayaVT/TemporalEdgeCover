# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:32:59 2021

@author: masahiro
"""


from scipy import sparse

from scipy.sparse import linalg

from BBt_list import create_BBt_list

#beta = 0.03
#delta = 0.04

### takes a list of square matrices expressing infection between
### nodes of the same type, probability of infection, of recover
### a list of edges to remove (edges between the same type nodes),
### and a list of nodes to remove (base idx of location nodes, e.g. 10000)
### should be subtracted.

### returns the leading eigenvalue of the system matrix

def computeThreshold(BBt_list, beta, delta, remove_edge=[], remove_node =[]):


    I = sparse.identity(BBt_list[0].shape[0])
    S = I.copy()
    I = I.tocsr()

    deltaI = (1-delta)*I
    for i in remove_node:
        deltaI[i,i] = 0

    for BBt in BBt_list:
        for edge in remove_edge:
            BBt[edge[0],edge[1]] = 0
        S = S@(deltaI+(beta**2)*BBt)

    lambda_S = linalg.eigs(S, k=1, tol=0.01, maxiter=100000)

    return abs(lambda_S[0][0])


### takes a list of bi-adjacency matrices, probability of infection, recover
### flag for whether B*B' or B'*B, a list of which nodes to remove, and scale
### in a list of nodes, base idx of location nodes, e.g. 10000 should be included 

### returns the leading eigenvalue of the system matrix and the system matrix itself
def Threshold_from_bi_ad_lists(bi_ad_list, beta, delta, which="m", remove_node =[], scale=1):

    if which=="m":
        I = sparse.identity(bi_ad_list[0].shape[0])
        S = I.copy()
        I = I.tocsr()
        deltaI = (1-delta)*I
        for i in remove_node:
            if i < 10000*scale:
                deltaI[i,i] = 0
        for i in range(len(bi_ad_list)-1):
            S = S@(deltaI+(beta**2)*(bi_ad_list[i]@bi_ad_list[i+1].transpose()))
        S = S@deltaI+(beta**2)*(bi_ad_list[-1]@bi_ad_list[0].transpose())

    else:
        I = sparse.identity(bi_ad_list[0].shape[1])
        S = I.copy()
        I = I.tocsr()
        deltaI = (1-delta)*I
        for i in remove_node:
            if i >= 10000*scale:
                deltaI[i-10000*scale,i-10000*scale] = 0
        for i in range(len(bi_ad_list)-1):
            S = S@(deltaI+(beta**2)*(bi_ad_list[i].transpose()@bi_ad_list[i+1]))
        S = S@(deltaI+(beta**2)*(bi_ad_list[-1].transpose()@bi_ad_list[0]))

    lambda_S = linalg.eigs(S, k=1, tol=0.01)
    return abs(lambda_S[0][0]), S
