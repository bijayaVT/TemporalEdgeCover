from read_ca_graph import read_ca_graph
from bi_list import create_biadjacency_list
from BBt_list import create_BBt_list
from threshold import Threshold_from_bi_ad_lists

###demo of computing lambda_S
###when remove edges, pass a list to "create_biadjacency_list"
###when remove nodes, pass a list to "Threshold_from_bi_ad_lists"

###beta=infection prob, deleta=recovery prob
###alpha=portion of nodes/edges to remove
###alpha=1 is deleting everything, alpha=0 is deleting nothing 

beta = 0.0185
delta = 0.04
alpha = 0.1

t,p,l, _, _ = read_ca_graph()

### load a pre-computed ordered list of nodes to remove
# def str_to_list_of_edges(aString):
#     a = aString[1:-1].split("), ")
#     for i in range(len(a)):
#         a[i] = tuple(map(int,a[i].strip("()").split(",")))
#     return a

# infile = open("ca_7d_gw_edges.txt")
# edges_to_remove = str_to_list_of_edges(infile.readline().strip())
# infile.close()

###generate edge deletion lambda
# bi_ad_list = create_biadjacency_list(t,p,l, remove_edge=edges_to_remove[:int(len(edges_to_remove)*alpha)])
# lam, S = Threshold_from_bi_ad_lists(bi_ad_list, beta, delta)


### load a pre-computed ordered list of nodes to remove
infile = open("ca_7d_gw_nodes.txt")
nodes_to_remove =  list(map(int, infile.readline().strip()[1:-1].split(", ")))
infile.close()

###generate node deletion lambda
bi_ad_list = create_biadjacency_list(t,p,l)
lam, S = Threshold_from_bi_ad_lists(bi_ad_list, beta, delta, remove_node=nodes_to_remove[:int(len(nodes_to_remove)*alpha)])

print(lam)
