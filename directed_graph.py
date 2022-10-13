# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 05:31:11 2021

@author: masahiro
"""

import networkx as nx
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
##This function takes temporal graphs (array of Graph objects), the number of
##people nodes (int 0<x<10000*scale), and the number of location nodes + 10000*scale (int x>10000*scale).
##Then it will return Directed Graph Object which is equivalent to F in the paper.

##Note that in the input graph object, poeople nodes are labeled as int from 0, and
##location nodes are labeled as int from 10000
###If the number of people nodes is > 10000 and set the base of location node
###to 100000, then use scale = 10

def directed_graph(t,p,l,scale=1):
    edges = []
    nodes = []

    for i in range(len(t)):
        for j in range(p):
            nodes.append(i*20000*scale+j)
        for k in range(l-10000*scale):
            nodes.append(i*20000*scale+k+10000*scale)

    for i in range(len(t)-1):
        for j in t[i].edges:
            if j[0] < 10000*scale:
                edges.append((i*20000*scale+j[0], i*20000*scale+j[1]))
            else:
                edges.append((i*20000*scale+j[1], i*20000*scale+j[0]))

        for k in t[i+1].edges:
            if k[0] >= 10000*scale:
                edges.append((i*20000*scale+k[0], (i+1)*20000*scale+k[1]))
            else:
                edges.append((i*20000*scale+k[1], (i+1)*20000*scale+k[0]))

    for j in t[-1].edges:
        if j[0] < 10000*scale:
            edges.append(((len(t)-1)*20000*scale+j[0], (len(t)-1)*20000*scale+j[1]))
        else:
            edges.append(((len(t)-1)*20000*scale+j[1], (len(t)-1)*20000*scale+j[0]))

    for k in t[0].edges:
        if k[0] >=  10000*scale:
            edges.append(((len(t)-1)*20000*scale+k[0], k[1]))
        else:
            edges.append(((len(t)-1)*20000*scale+k[1], k[0]))


    g = nx.DiGraph()

    g.add_edges_from(edges)
    g.add_nodes_from(nodes)

    return g


###This function converts DiGraph object to an adjacency matrix
def graph_to_mtx(g):

    nodes = list(g.nodes)
    nodes.sort()
    A = nx.adjacency_matrix(g, nodes).asfptype()

    return A


###This function takes X=F_{K−1}◦Fᵀ, the number of people (int 0<x<10000),
###and the number of locations (int 0<x<10000). Then it returns a matrix
###of shape num people by num locations where each element is
###the number of closed path of length k including the edge across all
###timesteps.

def block_wise_sum(prod, p,l, days, scale=1):
    if l>10000*scale:
        l-=10000*scale
    # print(type(prod))

    block_sum = np.zeros((p,l))

    c = 0
    for i in range(1,days):
        block_sum += prod[p*i+l*i:p*(i+1)+l*i, i*p+(i-1)*l:i*p+i*l]
        c+=1
        block_sum += prod[(i+1)*p+(i)*l:(i+1)*p+(i+1)*l, i*p+i*l:(i+1)*p+i*l].transpose()
        c+=1

    block_sum += prod[:p, days*p+(days-1)*l:]
    c+=1
    block_sum += prod[p:p+l, :p].transpose()
    c+=1
    # print(c)
    # print(block_sum.shape)

    # p1_l3 = prod[:p, 3*p+2*l:]
    # p2_l1= prod[p+l:2*p+l, p:p+l]
    # p3_l2 = prod[2*p+2*l:3*p+2*l, 2*p+l:2*p+2*l]

    # l1_p1 = prod[p:p+l, :p].transpose()
    # l2_p2 = prod[2*p+l:2*p+2*l, p+l:2*p+l].transpose()
    # l3_p3 = prod[3*p+2*l:, 2*p+2*l:3*p+2*l].transpose()


    # assert(p1_l3.shape == l1_p1.shape)
    # assert(p2_l1.shape == l1_p1.shape)

    # assert(l2_p2.shape == l1_p1.shape)
    # assert(p3_l2.shape == l1_p1.shape)

    # assert(l3_p3.shape == l1_p1.shape)

    # a = (p1_l3+l1_p1+p2_l1+l2_p2+p3_l2+l3_p3)
    # for i in range(p):
    #     for j in range(l):
    #         if int(block_sum[i,j]) != int(a[i,j]):
    #             print(i,j)

    block_sum = csr_matrix(block_sum)
    # assert(np.array_equal(block_sum,a))

    return block_sum


### This takes the "block sum" matrix computed from the previous function
### and returns arg max W−Score(i,j) and the max value.
def find_max(X):
    s,t = X.nonzero()

    # print(len(s))
    max_val = 0
    max_idx = 0

    for i in range(len(s)):
        val = X[s[i],t[i]]
        if val > max_val:
            max_val = val
            max_idx = i

    return s[max_idx],t[max_idx], max_val

import numpy as np
###This takes the "block sum" matrix from the block_wise_sum function,
###and returns the ranking of edges with respect to W−Score(i,j)

def find_topk(X):
    s,t = X.nonzero()

    print(len(s))
    lst = []
    for i in range(len(s)):
        lst.append(X[s[i],t[i]])

    idx = list(np.argsort(lst))
    idx.reverse()

    edges = []
    for j in idx:
        edges.append((s[j],t[j]))

    return edges

###This function takes temporal graphs (array of Graph objects), the number of
###people nodes (int 0<x<10000*scale), and the number of location nodes + 10000*scale (int x>10000*scale),
###The number of power k where k+1 is divisible (the number of timesteps)*2, and the scale depending on the size of the graphs
###Then it returns an ordered edge set with respect to W−Score(i,j) of one iteration in descending order.


def topk_walk(t,p,l,k=13, scale=1):
    g = directed_graph(t, p, l, scale)
    print("graph")
    A = graph_to_mtx(g)
    A_t = A.transpose()
    A_k = A
    for i in range(1,k):
        A_k = A_k@A


    prod = A_k.multiply(A_t)
    sum_p = block_wise_sum(prod, p, l, scale)

    sum_t = block_wise_sum(A_t, p, l, scale)

    X = sum_p + sum_t


    return find_topk(X)


###This function takes temporal graphs (array of Graph objects), the number of
###people nodes (int 0<x<10000*scale), the number of location nodes + 10000*scale (int x>10000*scale),
###int k where k+1 is divisible by the number of timesteps*2, and scale.

###Then it returns an ordered edge list based on Temporal Edge Cover.
def temporal_edge_cover(t,p,l,k=13, scale=1):

    edge_lst = []
    count = 0
    len_t = len(t)
    finished = 0

    nonzero = False
    X = None
    while nonzero==False:

        g = directed_graph(t, p, l, scale)

        A = graph_to_mtx(g)
        A_t = A.transpose()
        A_k = A
        for i in range(1,k):
            A_k = A_k@A




        prod = A_k.multiply(A_t)
        if len(prod.nonzero()[0])==0:
            print("break")
            # A_6 = A_5@A
            # spy_plot(A_6,p,l,"A_6 After Prod is 0")
            #spy_plot(A_5,p,l,"A_11 After Prod is 0")
            # spy_plot(A_5,p,l,"A_t After Prod is 0")
            # spy_plot(prod, p, l, "prod after prod is 0")
            # print(A_t.nonzero())
            # print("------------")
            # print(A_5.nonzero())
            nonzero = True
            if X != None:
                topk_edges = find_topk(X)
                topk_edges.remove(edge_lst[-1])
                edge_lst+=topk_edges


            break


        sum_p = block_wise_sum(prod, p, l, len_t, scale)
        sum_t = block_wise_sum(A_t, p, l, len_t, scale)

        X = sum_p + sum_t
        u,v,val = find_max(X)
        edge_lst.append((u,v))
        print("edge=",u,v)
        ### which timesteps contain the edge
        for i in range(len_t):
            if (u, v+10000*scale) in t[i].edges:
                print(i,"has the edge")
                t[i].remove_edge(u,v+10000*scale)

        print("========")
        print("count", len(edge_lst))
        print("=======")

    return edge_lst



###This function takes temporal graphs (array of Graph objects of length 3), the number of
###people nodes (int 0<x<10000), and the number of location nodes + 10000 (int x>10000).
###and int k where k+1 is divisible by the number of timesteps*2.
###Then it returns an ordered node list with respect to W−Score(i,i) of F^K of one iteration in descending order.
def node_gw_top_k(t,p,l, k=13, scale=1):

    len_t = len(t)
    g = directed_graph(t, p, l, scale)
    print("graph")
    A = graph_to_mtx(g)
    A_k = A
    for i in range(0,k):
        A_k = A_k@A

    I = identity((p+l-10000*scale)*len_t)
    # print(A_6.shape, I.shape)

    prod = I.multiply(A_6)


    block = np.zeros((p+l-10000*scale, p+l-10000*scale))
    for i in range(len_t):
        block += prod[i*p+i*(l-10000*scale):(i+1)*p+(i+1)*(l-10000*scale), i*p+i*(l-10000*scale):(i+1)*p+(i+1)*(l-10000*scale)]
    # block_sum = (prod[:p+l-10000, :p+l-10000]
    #                  + prod[ (p+l-10000) :(p+l-10000)*2, (p+l-10000):(p+l-10000)*2]
    #                  + prod[ ((p+l-10000)*2):, ((p+l-10000)*2):])
    # print("compare")
    # for i in range(p+l-10000):
    #     for j in range(p+l-10000):
    #         if int(block_sum[i,j]) != int(block[i,j]):
    #             print(i,j)


    values = np.diag(block)
    nodes = np.argsort(values*-1)

    node_list = []

    for node in nodes:
        if node >= p:
            node+=10000*scale-p
        node_list.append(node)

    return node_list


import numpy as np
from scipy.sparse import identity


##Takes the same input as temporal edge cover, but it returns
##an ordered node list based on the algorithm
def node_temporal_edge_cover(t,p,l,k=13, scale=1):
    print("temp edge cover")
    node_list = []
    count = 0
    len_t = len(t)
    finished = 0

    nonzero = False
    values = []
    while nonzero==False:

        g = directed_graph(t, p, l, scale)
        print("graph")
        A = graph_to_mtx(g)
        A_6 = A
        for i in range(0,k):
            A_6 = A_6@A

        I = identity((p+l-10000*scale)*len_t)

        # print(A_6.shape, I.shape)
        prod = I.multiply(A_6)

        print(prod.shape)
        if len(prod.nonzero()[0])==0:
            # print("break")
            # A_6 = A_5@A
            # spy_plot(A_6,p,l,"A_6 After Prod is 0")
            # spy_plot(A_6,p,l,"A_6 After Prod is 0")
            # spy_plot(A_5,p,l,"A_t After Prod is 0")
            # spy_plot(prod, p, l, "prod after prod is 0")
            # print(A_t.nonzero())
            # print("------------")
            # print(A_5.nonzero())
            # nonzero = True
            if len(values) != 0:
                topk_nodes = np.argsort(values*-1)
                for node in topk_nodes:
                    if node >= p:
                        node +=10000*scale-p
                    if node not in node_list:
                        node_list.append(node)
            break


        # print(type(prod))
        # block_sum = (prod[:p+l-10000, :p+l-10000]
        #              + prod[ (p+l-10000) :(p+l-10000)*2, (p+l-10000):(p+l-10000)*2]
        #              + prod[ ((p+l-10000)*2):, ((p+l-10000)*2):])

        # print(block_sum.shape, p, l-10000)

        # print(type(block_sum))


        block = np.zeros((p+l-10000*scale, p+l-10000*scale))
        for i in range(len_t):
            block += prod[i*p+i*(l-10000*scale):(i+1)*p+(i+1)*(l-10000*scale), i*p+i*(l-10000*scale):(i+1)*p+(i+1)*(l-10000*scale)]


        values = np.diag(block)
        node_removed = np.argmax(values)

        if node_removed >= p:
            node_removed+=10000*scale-p

        for i in range(len_t):
            t[i].remove_node(node_removed)
            t[i].add_node(node_removed)

        node_list.append(node_removed)

        print("========")
        print("count", len(node_list))
        print("=======")

    return node_list




import matplotlib.pyplot as plt
import time
def spy_plot(A,p,l,name="spy", days=7, scale=1):
    if l>10000*scale:
        l -= 10000
    if p > 10000*scale:
        p-=10000*scale

    fig, ax = plt.subplots(figsize=(30,30))
    ax.spy(A)

    plt.title(name,fontsize=100)

    # ax.axhline( linewidth=5)
    # ax.axhline(p, linewidth=5)
    # ax.axhline(p+l, linewidth=5)
    # ax.axhline(2*p+l, linewidth=5)
    # ax.axhline(2*p+2*l, linewidth=5)
    # ax.axhline(3*p+2*l, linewidth=5)
    # ax.axhline(3*p+3*l, linewidth=5)


    # if days==7:
    #         ax.axhline(4*p,3*l linewidth=5)
    #         ax.axhline(4*p, 4*l linewidth=5)
    #         ax.axhline(5*p+l, linewidth=5)
    #         ax.axhline(2*p+l, linewidth=5)
    #         ax.axhline(2*p+2*l, linewidth=5)
    #         ax.axhline(3*p+2*l, linewidth=5)
    #         ax.axhline(3*p+3*l, linewidth=5)

    for i in range(days):
        ax.axhline(p*i+l*i, linewidth=5)
        ax.axhline(p*(i+1)+l*i, linewidth=5)

        ax.axvline(p*i+l*i, linewidth=5)
        ax.axvline(p*(i+1)+l*i, linewidth=5)

        fig.canvas.draw()

        fig.canvas.flush_events()

        time.sleep(0.1)
        print(i)

    ax.axhline(p*days, l*days, linewidth=5)
    ax.axvline(p*days, days*l, linewidth=5)
    fig.canvas.draw()


    # ax.axvline( linewidth=5)
    # ax.axvline(p, linewidth=5)
    # ax.axvline(p+l, linewidth=5)
    # ax.axvline(2*p+l, linewidth=5)
    # ax.axvline(2*p+2*l, linewidth=5)
    # ax.axvline(3*p+2*l, linewidth=5)
    # ax.axvline(3*p+3*l, linewidth=5)



    name = name+".png"
    fig.savefig(name)

    plt.close()


# def directed_graph_with_weights(t,p,l,beta, delta):
#     edges_pl = []
#     nodes = []
#     edges_pp = []
#
#     for i in range(len(t)):
#         for j in range(p):
#             nodes.append(i*20000+j)
#         for k in range(l-10000):
#             nodes.append(i*20000+k+10000)
#
#
#     for i in range(len(t)-1):
#         for j in t[i].edges:
#             if j[0] < 10000:
#                 edges_pl.append((i*20000+j[0], i*20000+j[1]))
#         for k in t[i+1].edges:
#             if k[0] > 10000:
#                 edges_pl.append((i*20000+j[1], (i+1)*20000+j[0]))
#
#         for node in t[i].nodes:
#             edges_pp.append((i*20000+node,(i+1)*20000+node))
#
#     for j in t[-1].edges:
#         if j[0] < 10000:
#             edges_pl.append(((len(t)-1)*20000+j[0], (len(t)-1)*20000+j[1]))
#
#     for k in t[0].edges:
#         if k[0] >  10000:
#             edges_pl.append(((len(t)-1)*20000+j[1], j[0]))
#
#     for node in t[-1].nodes:
#             edges_pp.append(((len(t)-1)*20000+node,node))
#
#
#     g = nx.Graph()
#
#     g.add_edges_from(edges_pl, weight=beta)
#     g.add_edges_from(edges_pp, weight=1-delta)
#     g.add_nodes_from(nodes)
#
#     return g

def lambda_directed_g(g):
    A = nx.adjacency_matrix(g).asfptype()
    lambda_A = linalg.eigs(A, k=1)

    return abs(lambda_A[0][0])
