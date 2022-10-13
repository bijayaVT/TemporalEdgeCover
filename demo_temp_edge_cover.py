from directed_graph import temporal_edge_cover, node_temporal_edge_cover
from read_ca_graph import read_ca_graph

###demo of temporal edge cover

### ordered_edges_list contains a list of (people_node_idx, location_node_idx)
### note that 10000*scale is not added to the location_node_idx

### ordered_nodes_list contains mix of people_node_idx and location_node_idx
### note that 10000*scale is included in the location nodes to distinguish
### them from the people nodes 

t, p, l, _, _ = read_ca_graph()
ordered_edge_list = temporal_edge_cover(t,p,l)

# ordered_nodes_list =node_temporal_edge_cover(t,p,l)
