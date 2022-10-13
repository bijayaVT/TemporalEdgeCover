# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:00:29 2022

@author: masahiro
"""
import sys
sys.path.append("../")
import networkx as nx


###read Carilion files and create a list of graph objects
###return a list of graphs, people counts, location counts (10000 is a base index of
###location nodes to distinguish people and location nodes), mappings from
###node ids of the graph ojects to ids of the original Carilion file

###Note that if the number of poeple node is > 10000, please set the base of
###loc nodes to 100000
def read_ca_graph():
    Temporalgraph = []

    days = []
    people_nodes = {}
    location_nodes = {}

    people_counter = 0
    location_counter = 10000

    line_counter = 0

    people_idx_to_id = {}
    location_idx_to_id = {}

    need_nodes = open("unnecessary_nodes", "r")
    nodes = need_nodes.readline().strip().strip("[ ]").split(",")
    nodes = list(map(int, nodes))
    need_nodes.close()


    for i in range(7):
        filename = "ca_graph_files/HAISE_Carilion_all_visits_day_"+str(i)+".txt"
        G = nx.Graph()
        Temporalgraph.append(G)
        days.append(i)

        for line in open(filename):
            if line_counter > 0:

                line = line.strip()
                line = line.split()

                if int(line[0]) not in nodes or int(line[1]) not in nodes:
                    continue

                if line[0] not in people_nodes:
                    people_nodes[line[0]] = people_counter
                    people_idx_to_id[people_counter] = int(line[0])
                    person = people_nodes[line[0]]
                    people_counter += 1


                person = people_nodes[line[0]]

                if line[1] not in location_nodes:
                    location_nodes[line[1]] = location_counter
                    location_idx_to_id[location_counter-10000] = int(line[1])
                    location_counter += 1
                room = location_nodes[line[1]]

                G.add_edge(person, room)
            line_counter += 1
        line_counter = 0

    for graph in Temporalgraph:
        graph.add_nodes_from(range(people_counter))
        graph.add_nodes_from(range(10000,location_counter))

        # print(graph.number_of_nodes(), graph.number_of_edges())

    print("people", people_counter, "locations",location_counter-10000)


    return Temporalgraph, people_counter, location_counter, people_idx_to_id, location_idx_to_id
