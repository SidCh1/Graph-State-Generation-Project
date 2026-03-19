#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:55:57 2025

@author: siddhu
"""

import matplotlib.pyplot as plt
import networkx as nx

# import numpy as np
import random

# from collections import defaultdict, deque
from networkx.drawing.layout import circular_layout
from itertools import combinations, groupby
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd

from networkx.algorithms.approximation import dominating_set

# import math
# import csv
# import concurrent.futures
# import time
# from multiprocessing import Pool, cpu_count
# from joblib import Parallel, delayed
# import pickle
# import joblib


def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    # print(G.edges())
    return G


def BA(n, c, alt=False):
    """
    Generates a random sample from the Barabasi-Albert ensemble,
    starting from the complete graph of c+1 nodes, such that
    the total number of nodes in the graph is n=c+1+x, with
    x=0,1,2,...

    Note that n should be greater than or equal to c+1.
    """

    if alt:
        ### Here, the starting graph is the star graph of c+1 nodes
        return nx.barabasi_albert_graph(n, c)
    else:
        return nx.barabasi_albert_graph(n, c, initial_graph=nx.complete_graph(c + 1))


def draw_graph(graph, color="orange", layout="circular"):
    """
    Draw the given graph using NetworkX and Matplotlib.
    """
    if layout == "circular":
        pos = circular_layout(graph)
    else:
        pos = nx.spring_layout(
            graph
        )  # Default to spring layout if layout is not circular

    plt.figure(figsize=(5, 5))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=color,
        node_size=300,
        font_size=10,
        font_color="black",
        font_weight="bold",
    )
    # plt.title("Graph Visualization")
    plt.show()


def pick_stars_ss(G):
    MG = {}
    SG = {}

    tn = set()  # to track all nodes used so far
    MG[0] = G.copy()
    all_nodes = set(MG[0].nodes())

    # Step 1: Create a sorted list of nodes by degree (descending)
    degree_table = sorted(MG[0].degree(), key=lambda x: x[1], reverse=True)
    # print("Degree Table (sorted):", degree_table)

    # Step 2: Iterate through the sorted list to build stars
    for l, (center_node, _) in enumerate(degree_table):
        # if center_node in tn:
        #     continue  # skip if already covered

        neighbors = list(MG[0].neighbors(center_node))

        # Build strict star: only center connected to each neighbor
        SG[l] = nx.Graph()
        SG[l].add_node(center_node)
        for neighbor in neighbors:
            SG[l].add_edge(center_node, neighbor)

        # Update covered nodes
        tn.update([center_node] + neighbors)

        # Stop if all nodes are covered
        if tn >= all_nodes:
            break

    return SG

# def pick_stars_ss(G):
#     MG = {}
#     SG = {}
#     OPSG = {}
#     sid_total_gates = 0
#     # total_gates = len(connected_subgraph)-2 # gives the no of nodes,
#     tn = []
#     edu = []

#     MS = []  # a list to store all small stars and merging sequence.

#     l = 0

#     MG[0] = G.copy()
#     an = list(MG[0].nodes())

#     while MG[l].size() > 0:
#         a = max(dict(MG[l].degree()).items(), key=lambda x: x[1])
#         b = MG[l].edges(a[0])
#         SG[l] = nx.Graph(b)
#         MG[l + 1] = MG[l].copy()
#         MG[l + 1].remove_node(a[0])
#         # draw_graph(MG[l], 'yellow', layout='circular')
#         # draw_graph(SG[l], 'pink', layout='circular')
#         tn.extend(list(SG[l].nodes()))
#         edu.extend(list(SG[l].edges()))

#         check = all(item in tn for item in an)
#         if check is True:
#             break
#         else:
#             l += 1

#     return SG


def connected_stars(G):  #:, SG):
    """
    CSG means Connected Sub Graphs.
    SGcp = copy of SG, Sub Graphs (those sub graphs are stars picked in above function pick_stars_ss)
    """
    SG = pick_stars_ss(G)

    if len(SG) == 1:
        CSG_list = list(SG.values())
        # print("I am here to exit abce")

    else:
        CSG = {}

        # Make a copy of SG
        SGcp = SG.copy()

        # Counter for sequential indices in CSG
        csg_index = 0

        # Create a list of keys to iterate over
        keys_to_delete = []

        # Iterate through each subgraph in SGcp
        for sg_key, sg in SGcp.items():
            """
            This for loop is just to push two stars which share a common node(s) from SGCP into CSG. If no two such two stars are found then extra edge between those disjoint stars is also added to CSG along witht the corresponding two stars. Once added to CSG, those exact stars from SGcp are deleted. So CSG will have stars that are defintely connected. 
            """

            # Initialize common_found flag for current subgraph
            common_found = False

            # Iterate through all remaining subgraphs to find common nodes
            for i, other_sg in SGcp.items():
                if sg_key == i:
                    continue  # Skip comparing the same subgraph with itself

                common_nodes = set(sg.nodes()).intersection(set(other_sg.nodes()))
                if common_nodes:
                    # print(f"Common nodes found between sg[{sg_key}] and sg[{i}]: {common_nodes}")
                    common_found = True
                    # Append both subgraphs to CSG
                    CSG[csg_index] = sg
                    csg_index += 1
                    CSG[csg_index] = other_sg
                    csg_index += 1
                    # Mark keys for deletion
                    keys_to_delete.extend([sg_key, i])
                    # Break out of the loop since common nodes are found
                    break

            if common_found:
                break  # Stop iterating over other subgraphs if common nodes are found

            if not common_found:
                # print(f"No common nodes found for sg[{sg_key}], so going to find an edge.")

                # Check for an edge between SGcp[0] and other subgraphs
                for i, other_sg in SGcp.items():
                    if i == 0:
                        continue  # Skip comparing with SGcp[0]

                    # Check for an edge between SGcp[0] and other_sg
                    for u, v in G.edges:
                        if (u in sg.nodes and v in other_sg.nodes) or (
                            v in sg.nodes and u in other_sg.nodes
                        ):
                            # print(f"Edge found between sg[0] and sg[{i}]: ({u}, {v})")
                            # Add the edge as a graph to CSG
                            edge_graph = nx.Graph()
                            edge_graph.add_edge(u, v)
                            CSG[csg_index] = edge_graph
                            csg_index += 1
                            # Add SGcp[0] and other_sg to CSG
                            CSG[csg_index] = sg
                            csg_index += 1
                            CSG[csg_index] = other_sg
                            csg_index += 1
                            # Mark keys for deletion
                            keys_to_delete.extend([0, i])
                            common_found = True
                            break  # Stop searching for edges once a match is found

                    if common_found:
                        break  # Stop searching for edges if a match is found

                if common_found:
                    break  # Stop iterating over subgraphs if an edge is found

        # Delete marked keys from SGcp
        for key in keys_to_delete:
            del SGcp[key]

        SGcp_list = list(
            SGcp.values()
        )  # There is some prob to continue with dicts so converted to list.
        CSG_list = list(
            CSG.values()
        )  # There is some prob to continue with dicts so converted to list.
        # print("I am out of first loop")
        # for i, graph in enumerate(CSG_list):
        #     draw_graph(graph, 'lightblue', layout='circular')

        """
        Now we have at least two stars in CSG. Those are deleted from SGcp. Now we do two more loops called part 1 and part 2 to push the rest of the SGcp stars to the connected stars list CSG. 
        
        The code below is of two parts. part 1 is to check if there is a common node between any graphs SGcp and CSG, if yes those are pushed from SGcp to CSG. 
        The part 2 is when all graphs are disjoint. Then it finds a common edge between any SGcp and CSG graph and adds that to SGcp. 
        
        Part 1 and 2 are excuted untill the SGcp list is empty. 
        """

        execute_part_1 = True

        while SGcp_list:
            # print("Entering while loop")
            # print("Length of SGcp_list:", len(SGcp_list))
            # print("Length of CSG_list:", len(CSG_list))
            # for sgcp_graph in SGcp_list:
            # print("Nodes in sgcp_graph:", sgcp_graph.nodes())

            if execute_part_1:
                # print("Executing Part 1")
                for sgcp_graph in SGcp_list:
                    for csg_graph in CSG_list:
                        common_nodes = set(sgcp_graph.nodes()).intersection(
                            csg_graph.nodes()
                        )
                        if common_nodes:
                            # print("Found common nodes")
                            # Copy the graph from SGcp_list to CSG_list
                            CSG_list.append(sgcp_graph.copy())
                            # print("Length of CSG_list after appending:", len(CSG_list))
                            # Remove the graph from SGcp_list
                            SGcp_list.remove(sgcp_graph)
                            # print("Length of SGcp_list after removing:", len(SGcp_list))
                            break  # Break the inner loop as we found a match for the current sgcp_graph
                execute_part_1 = False  # Switch to Part 2 for the next iteration
            if SGcp_list:  # Check if SGcp_list is not empty before entering Part 2
                # print("Executing Part 2")
                # Flag to track if a connecting edge is found
                connecting_edge_found = False

                for sgcp_graph in SGcp_list:
                    # Iterate through each node in sgcp_graph
                    # draw_graph(sgcp_graph, 'lightgreen', layout='circular')
                    for node in sgcp_graph.nodes():
                        # Check if the node has an edge connecting to any graph in CSG_list
                        for csg_graph in CSG_list:
                            # Iterate over edges of the original graph G
                            for edge in G.edges():
                                # Check if the edge is between the current node in sgcp_graph and any node in csg_graph
                                if (
                                    edge[0] in sgcp_graph.nodes()
                                    and edge[1] in csg_graph.nodes()
                                ) or (
                                    edge[1] in sgcp_graph.nodes()
                                    and edge[0] in csg_graph.nodes()
                                ):
                                    # print("Found connecting edge", edge)
                                    # Create a new graph with only the connecting edge
                                    new_graph = nx.Graph()
                                    new_graph.add_edge(
                                        *edge
                                    )  # Add the edge to the new graph

                                    # Add the new graph to CSG_list
                                    CSG_list.append(new_graph)
                                    # print("Length of CSG_list after appending:", len(CSG_list))

                                    # Add the original sgcp_graph to CSG_list as well
                                    CSG_list.append(sgcp_graph)

                                    # Remove the original sgcp_graph from SGcp_list
                                    SGcp_list.remove(sgcp_graph)
                                    # print("Length of SGcp_list after removing:", len(SGcp_list))

                                    # Set the flag to True to indicate a connecting edge is found
                                    connecting_edge_found = True

                                    # Exit all loops
                                    break

                            if connecting_edge_found:
                                break  # Exit the inner loop if a connecting edge is found
                        if connecting_edge_found:
                            break  # Exit the outer loop if a connecting edge is found
                    if connecting_edge_found:
                        break  # Exit the outermost loop if a connecting edge is found

                execute_part_1 = True  # Switch back to Part 1 for the next iteration

            else:
                # print("SGcp_list is empty, breaking the loop")
                break  # Exit the loop if SGcp_list is empty

    return CSG_list


def calculate_gate_ss(G):  # ,CSG_list):
    """
    Calculates the gates using the SS method on the given graph G.
    """

    if not G.edges():  ### If there are no edges in the graph
        return 0, 0, 0, 0
    else:
        CSG_list = connected_stars(G)

        MS = []
        SG = {index: value for index, value in enumerate(CSG_list)}
        sid_total_gates = 0

        MS.append(SG[0].copy())
        merge_impossible = False

        while len(SG) > 1:
            common_node_found = False

            common_node_index = 0

            for i in SG.keys():
                # print(i)
                if i == 0:
                    continue

                nodes_SG_0 = set(SG[0].nodes())
                nodes_SG_i = set(SG[i].nodes())

                common_nodes_sg0_sg_i = nodes_SG_0 & nodes_SG_i

                if common_nodes_sg0_sg_i:
                    common_node_found = True
                    common_node_index = i

                    if (
                        nodes_SG_0 == nodes_SG_i
                        or nodes_SG_i.issubset(nodes_SG_0)
                        or nodes_SG_0.issubset(nodes_SG_i)
                    ):
                        # If all nodes in SG[i] are in SG[0] or vice versa, exit the function as one graph is included in another
                        pass

                    else:
                        # common_nodes = list(set(SG[0].nodes()) & set(SG[i].nodes()))
                        common_nodes = list(nodes_SG_0 & nodes_SG_i)

                        # Choose common node
                        common_node = None

                        max_degree_node_sg0 = max(
                            dict(SG[0].degree()).items(), key=lambda x: x[1]
                        )[0]
                        max_degree_node_sgi = max(
                            dict(SG[i].degree()).items(), key=lambda x: x[1]
                        )[0]

                        for node in common_nodes:
                            if (
                                node == max_degree_node_sg0
                                and node == max_degree_node_sgi
                            ):
                                common_node = node
                                break

                        if common_node is None:
                            for node in common_nodes:
                                if node == max_degree_node_sgi:
                                    common_node = node
                                    break

                        if common_node is None:
                            for node in common_nodes:
                                if node == max_degree_node_sg0:
                                    common_node = node
                                    break

                        if common_node is None:
                            common_node = random.choice(common_nodes)

                        # Shifting center of SG[0]: If condition is when we dont have to shift the center and therefore no gates.
                        # Else case is when we need to shift and we count gates.
                        if (
                            common_node
                            == max(dict(SG[0].degree()).items(), key=lambda x: x[1])[0]
                            or len(SG[0].nodes()) == 2
                        ):
                            pass

                        else:
                            SG[0].remove_edges_from(SG[0].edges())
                            # Add edges from common_node to all other nodes in SG[0] (avoid self-edges)
                            for node in SG[0].nodes():
                                if node != common_node:
                                    SG[0].add_edge(
                                        common_node, node, weight=1
                                    )  # Adding edge weight of 1

                        # Delete all common nodes in SG[i] except the chosen common node and center
                        for node in common_nodes:
                            max_degree_node = max(
                                dict(SG[i].degree()).items(), key=lambda x: x[1]
                            )[0]
                            if node != common_node and node != max_degree_node:
                                SG[i].remove_node(node)
                        MS.append(SG[i].copy())  # To store all the updated Stars

                        # Shifting center of SG[i]: If condition is when we dont have to shift the center and therefore no gates.
                        # Else case is when we need to shift and we count gates.
                        if (
                            common_node
                            == max(dict(SG[i].degree()).items(), key=lambda x: x[1])[0]
                            or len(SG[i].nodes()) == 2
                        ):
                            pass

                        else:
                            # Add edges from common node in SG[0] to all nodes in SG[i] (avoid self-edges)
                            SG[i].remove_edges_from(SG[i].edges())
                            for node in SG[i].nodes():
                                if node != common_node:
                                    SG[i].add_edge(
                                        common_node, node, weight=1
                                    )  # Adding edge weight of 1

                        # Merging the two stars
                        for node in SG[i].nodes():
                            if node != common_node:
                                SG[0].add_edge(
                                    common_node, node, weight=1
                                )  # Adding edge weight of 1

                        sid_total_gates += 1  # For merging (one CNOT gate)

                    break

            if common_node_found:
                del SG[common_node_index]  # Remove the subgraph at index i

            if not common_node_found:
                merge_impossible = True
                print("stars are disjoint, merge impossible")
                print("merge_impossible is", merge_impossible)
                break  # No common nodes found, exit the loop

        # Find total Bell pairs used
        total_edges = 0
        for graph in MS:
            total_edges += graph.number_of_edges()

        # Adding Star formation gates
        for pq in range(len(MS)):
            if MS[pq].number_of_nodes() == 2:
                pass
            else:
                sid_total_gates += MS[pq].number_of_nodes() - 2

        return sid_total_gates, MS, total_edges, merge_impossible


def mst_and_internal_nodes(G):
    """
    Compute the Minimum Spanning Tree (MST) of a graph and return:
    1. The MST graph
    2. The number of internal nodes (degree >= 2) in the MST

    Parameters:
        G (networkx.Graph): Input weighted undirected graph

    Returns:
        mst (networkx.Graph): Minimum Spanning Tree of G
        num_internal_nodes (int): Number of internal nodes in the MST
    """
    # Compute the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)

    # Count internal nodes
    internal_nodes = [node for node in mst.nodes if mst.degree[node] >= 2]
    num_internal_nodes = len(internal_nodes)

    return mst, num_internal_nodes


def merge_subgraphs(subgraphs):
    """
    Merge a list of NetworkX graphs into a single graph containing all nodes and edges.

    Parameters:
        subgraphs (list of networkx.Graph): List of subgraphs to merge.

    Returns:
        networkx.Graph: A single graph that contains all nodes and edges from the subgraphs.
    """
    return nx.compose_all(subgraphs)


def check_tree_and_internal_nodes(G):
    """
    Check if a graph is a tree. If so, return True and the number of internal nodes.
    Otherwise, return False and None.

    Parameters:
        G (networkx.Graph): The graph to check.

    Returns:
        (bool, int or None): Tuple of (is_tree, number_of_internal_nodes or None)
    """
    if nx.is_tree(G):
        internal_nodes = [node for node in G.nodes if G.degree[node] >= 2]
        return True, len(internal_nodes)
    else:
        return False, None
    
    
def get_min_dominating_set_size(G):
    """
    Compute the size of a minimum dominating set of a graph G.

    Parameters:
        G (networkx.Graph): Input graph.

    Returns:
        int: Size of a minimum dominating set.
    """
    dom_set = dominating_set.min_weighted_dominating_set(G)
    return len(dom_set)




# ------------------------------
# Core simulation logic
# ------------------------------
def run_single_simulation(N):
    c = max(1, round(0.05 * N))  # Avoid c=0
    G = nx.barabasi_albert_graph(N, c)

    # Run SS protocol once and reuse outputs
    gates, MS, _, _ = calculate_gate_ss(G)

    # Merged subgraph from star-based protocol
    merged_graph = merge_subgraphs(MS)

    # Dominating set size from merged graph
    ss_mds_size = get_min_dominating_set_size(merged_graph)

    # Internal nodes from SS merged graph
    _, ss_int_nodes = check_tree_and_internal_nodes(merged_graph)

    # MST and its internal nodes
    mst, mst_int_nodes = mst_and_internal_nodes(G)

    # Dominating set size from MST
    mst_mds_size = get_min_dominating_set_size(mst)

    return {
        "N": N,
        "sources": len(MS),
        "ss_internal_nodes": ss_int_nodes,
        "mst_internal_nodes": mst_int_nodes,
        "ss_mds_size": ss_mds_size,
        "mst_mds_size": mst_mds_size,
    }


# ------------------------------
# Wrapper for multiprocessing
# ------------------------------
def run_simulation_wrapper(args):
    return run_single_simulation(*args)


# ------------------------------
# Main simulation driver
# ------------------------------
def run_simulations():
    Ns = list(range(100, 501, 20))  # N from 100 to 500
    num_runs = 500  # Adjust if needed

    all_tasks = [(N,) for N in Ns for _ in range(num_runs)]

    with ProcessPoolExecutor() as executor:
        futures = list(
            tqdm(executor.map(run_simulation_wrapper, all_tasks), total=len(all_tasks))
        )

    df = pd.DataFrame(futures)
    df_mean = df.groupby("N").mean().reset_index()
    df_mean.to_csv("simulation_statistics_vs_N_BA_005_500_N2.csv", index=False)
    print("Saved results to simulation_statistics_vs_N_ba.csv")


# ------------------------------
# Run it
# ------------------------------
if __name__ == "__main__":
    run_simulations()