# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:38:06 2021

@author: masahiro
"""


from networkx.algorithms import bipartite
import networkx as nx
from scipy.sparse import csr_matrix

###Takes a list of graph objects, poeple counter, location counter, and a list of which edges to remove
###Return a list of bi-adjacency matrices. 

def create_biadjacency_list(Temporalgraph, num_p, num_l, remove_edge=[], remove_nodes=[], scale=1):
    bi_adjacency_list = []
    for graph in Temporalgraph:
        edges = set(graph.edges)

        graph.remove_nodes_from(remove_nodes)
        bi = bipartite.biadjacency_matrix(graph, list(range(num_p)), list(range(10000*scale,num_l)))
        for edge in remove_edge:
            bi[edge[0],edge[1]] = 0
        bi_adjacency_list.append(bi)
        graph.add_edges_from(edges)
        graph.add_nodes_from(remove_nodes)
    return bi_adjacency_list
