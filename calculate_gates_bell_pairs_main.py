#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:02:32 2024

@author: siddhu
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from collections import defaultdict, deque
from networkx.drawing.layout import circular_layout
from itertools import combinations, groupby
import math
import csv
import concurrent.futures
import time
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import pickle


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


def generate_data_BA(num_samples=500, display=True, save_to_file=True):

    # N=np.linspace(50,500,10,dtype=int)
    # N=np.linspace(50,200,1,dtype=int)
    N = [100, 200, 300]
    P = np.arange(0.01, 0.98, 0.02)
    F = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    print(
        "Generating data for BA networks with number of nodes N =",
        N,
        "and fractions P =",
        P,
    )
    print("\nThe number of samples is", num_samples)

    C = {}  ### size of the largest connected component
    R = {}  ### density of the full graph
    R0 = {}  ### density of the largest connected component
    K = {}  ### max degree of the full graph
    K0 = {}  ### max degree of the largest connected component
    Cl = {}  ### average clustering coefficient of the full graph
    Cl0 = {}  ### average clustering coefficient of the largest connected component
    deg = {}  ### average degree of the full graph
    deg0 = {}  ### average degree of the largest connected component
    gates_SS = {}
    stars_SS = {}
    gates_MMG = {}
    gates_MMGupd = {}

    for n in N:
        for f in F:
            for p in P:
                C[n, f, p] = []
                R[n, f, p] = []
                R0[n, f, p] = []
                K[n, f, p] = []
                K0[n, f, p] = []
                Cl[n, f, p] = []
                Cl0[n, f, p] = []
                deg[n, f, p] = []
                deg0[n, f, p] = []
                gates_SS[n, f, p] = []
                stars_SS[n, f, p] = []
                gates_MMG[n, f, p] = []
                gates_MMGupd[n, f, p] = []

    def take_sample(n, p):  ### When in the complete case (f=1)
        # print(0)
        c = int(n * p)
        if c == 0:
            c = 1
        G = BA(n, c)
        # S = sorted(nx.connected_components(G), key=len, reverse=True)
        # G0 = G.subgraph(S[0])  ### The largest connected component
        steiner_G0 = generate_steiner_subgraph(G, list(G.nodes()))
        # print('done generating graphs')

        data = {}
        data["size"] = len(G.nodes())
        data["density"] = nx.density(G)
        # data['density of largest component']=nx.density(G0)
        data["highest degree"] = max(degree_distribution(G).values())
        # data['highest degree of largest component']=max(degree_distribution(G0).values())
        data["average degree"] = average_degree(G)
        # data['average degree of largest component']=average_degree(G0)
        data["average clustering coefficient"] = nx.average_clustering(G)
        # data['average clustering coefficient of largest component']=nx.average_clustering(G0)
        ss_output = calculate_gate_ss(G)
        data["gates SS"] = ss_output[0] / n
        data["number of stars"] = len(ss_output[1])
        # print('done ss')
        data["gates MMG"] = calculate_gate_steiner(G, list(G.nodes())) / n
        # print('done MMG')
        data["gates MMG upd"] = calculate_gate_ss(steiner_G0)[0] / n
        # print('done MMG upd')

        return data

    def take_sample_subset(n, f, p):
        # print(0)
        c = int(n * p)
        if c == 0:
            c = 1
        G = BA(n, c)
        m = int(np.ceil(n * f))
        S = generate_selected_nodes(G, m)
        G0 = generate_connected_subgraph_sid(G, S)
        steiner_G0 = generate_steiner_subgraph(G, S)

        data = {}
        data["size"] = len(G0.nodes())
        data["density"] = nx.density(G)
        data["density of subset"] = nx.density(G0)
        data["highest degree"] = max(degree_distribution(G).values())
        data["highest degree of subset"] = max(degree_distribution(G0).values())
        data["average degree"] = average_degree(G)
        data["average degree of subset"] = average_degree(G0)
        data["average clustering coefficient"] = nx.average_clustering(G)
        data["average clustering coefficient of subset"] = nx.average_clustering(G0)
        ss_output = calculate_gate_ss(G0)
        data["gates SS"] = ss_output[0] / m
        data["number of stars"] = len(ss_output[1])
        # print('done ss')
        data["gates MMG"] = calculate_gate_steiner(G0, list(G0.nodes())) / m
        # print('done MMG')
        data["gates MMG upd"] = calculate_gate_ss(steiner_G0)[0] / m
        # print('done MMG upd')

        return data

    output = {}

    for n in N:
        for p in P:
            for f in F:
                if display:
                    print(n, f, p)
                if f == 1:
                    output[n, f, p] = Parallel(n_jobs=8)(
                        delayed(take_sample)(n, p) for i in range(num_samples)
                    )
                    # output[n,d] = [take_sample(n,d) for i in range(num_samples)]

                    C[n, f, p].append(
                        np.mean([data["size"] for data in output[n, f, p]])
                    )
                    R[n, f, p].append(
                        np.mean([data["density"] for data in output[n, f, p]])
                    )
                    # R0[n,f].append(np.mean([data['density of subset'] for data in output[n,f,p]]))
                    K[n, f, p].append(
                        np.mean([data["highest degree"] for data in output[n, f, p]])
                    )
                    # K0[n,f].append(np.mean([data['highest degree of subset'] for data in output[n,f,p]]))
                    Cl[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    # Cl0[n,f].append(np.mean([data['average clustering coefficient of subset'] for data in output[n,f,p]]))
                    deg[n, f, p].append(
                        np.mean([data["average degree"] for data in output[n, f, p]])
                    )
                    # deg0[n,f].append(np.mean([data['average degree of subset'] for data in output[n,f,p]]))
                    gates_SS[n, f, p].append(
                        np.mean([data["gates SS"] for data in output[n, f, p]])
                    )
                    stars_SS[n, f, p].append(
                        np.mean([data["number of stars"] for data in output[n, f, p]])
                    )
                    gates_MMG[n, f, p].append(
                        np.mean([data["gates MMG"] for data in output[n, f, p]])
                    )
                    gates_MMGupd[n, f, p].append(
                        np.mean([data["gates MMG upd"] for data in output[n, f, p]])
                    )
                else:
                    output[n, f, p] = Parallel(n_jobs=8)(
                        delayed(take_sample_subset)(n, f, p) for i in range(num_samples)
                    )
                    # output[n,d] = [take_sample(n,d) for i in range(num_samples)]

                    C[n, f, p].append(
                        np.mean([data["size"] for data in output[n, f, p]])
                    )
                    R[n, f, p].append(
                        np.mean([data["density"] for data in output[n, f, p]])
                    )
                    R0[n, f, p].append(
                        np.mean([data["density of subset"] for data in output[n, f, p]])
                    )
                    K[n, f, p].append(
                        np.mean([data["highest degree"] for data in output[n, f, p]])
                    )
                    K0[n, f, p].append(
                        np.mean(
                            [
                                data["highest degree of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    Cl[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    Cl0[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    deg[n, f, p].append(
                        np.mean([data["average degree"] for data in output[n, f, p]])
                    )
                    deg0[n, f, p].append(
                        np.mean(
                            [
                                data["average degree of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    gates_SS[n, f, p].append(
                        np.mean([data["gates SS"] for data in output[n, f, p]])
                    )
                    stars_SS[n, f, p].append(
                        np.mean([data["number of stars"] for data in output[n, f, p]])
                    )
                    gates_MMG[n, f, p].append(
                        np.mean([data["gates MMG"] for data in output[n, f, p]])
                    )
                    gates_MMGupd[n, f, p].append(
                        np.mean([data["gates MMG upd"] for data in output[n, f, p]])
                    )

    output_avg = {}
    for n in N:
        for f in F:
            for p in P:
                output_avg[n, f, p] = {}
                output_avg[n, f, p]["size"] = np.array(C[n, f, p])
                output_avg[n, f, p]["density"] = np.array(R[n, f, p])
                output_avg[n, f, p]["density of subset"] = np.array(R0[n, f, p])
                output_avg[n, f, p]["highest degree"] = np.array(K[n, f, p])
                output_avg[n, f, p]["highest degree of subset"] = np.array(K0[n, f, p])
                output_avg[n, f, p]["average clustering coefficient"] = np.array(
                    Cl[n, f, p]
                )
                output_avg[n, f, p]["average clustering coefficient of subset"] = (
                    np.array(Cl0[n, f, p])
                )
                output_avg[n, f, p]["average degree"] = np.array(deg[n, f, p])
                output_avg[n, f, p]["average degree of subset"] = np.array(
                    deg0[n, f, p]
                )
                output_avg[n, f, p]["gates SS"] = np.array(gates_SS[n, f, p])
                output_avg[n, f, p]["number of stars"] = np.array(stars_SS[n, f, p])
                output_avg[n, f, p]["gates MMG"] = np.array(gates_MMG[n, f, p])
                output_avg[n, f, p]["gates MMG upd"] = np.array(gates_MMGupd[n, f, p])

    if save_to_file:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "data_BA_" + timestr + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump([N, F, P, num_samples, output_avg, output], f)

        return N, F, P, num_samples, output_avg, output

    else:
        return N, F, P, num_samples, output_avg, output


def generate_data_ER(num_samples=500, display=True, save_to_file=True):

    # N=np.linspace(50,500,10,dtype=int)
    # N=np.linspace(50,200,1,dtype=int)
    N = [100, 200, 300]
    P = np.arange(0.01, 0.98, 0.02)
    F = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    print(
        "Generating data for ER networks with number of nodes N =",
        N,
        "and probabilities P =",
        P,
    )
    print("\nThe number of samples is", num_samples)

    C = {}  ### size of the largest connected component
    R = {}  ### density of the full graph
    R0 = {}  ### density of the largest connected component
    K = {}  ### max degree of the full graph
    K0 = {}  ### max degree of the largest connected component
    Cl = {}  ### average clustering coefficient of the full graph
    Cl0 = {}  ### average clustering coefficient of the largest connected component
    deg = {}  ### average degree of the full graph
    deg0 = {}  ### average degree of the largest connected component
    gates_SS = {}
    stars_SS = {}
    gates_MMG = {}
    gates_MMGupd = {}

    for n in N:
        for f in F:
            for p in P:
                C[n, f, p] = []
                R[n, f, p] = []
                R0[n, f, p] = []
                K[n, f, p] = []
                K0[n, f, p] = []
                Cl[n, f, p] = []
                Cl0[n, f, p] = []
                deg[n, f, p] = []
                deg0[n, f, p] = []
                gates_SS[n, f, p] = []
                stars_SS[n, f, p] = []
                gates_MMG[n, f, p] = []
                gates_MMGupd[n, f, p] = []

    def take_sample(n, p):  ### When in the complete case (f=1)
        # print(0)
        G = gnp_random_connected_graph(n, p)
        # S = sorted(nx.connected_components(G), key=len, reverse=True)
        # G0 = G.subgraph(S[0])  ### The largest connected component
        steiner_G0 = generate_steiner_subgraph(G, list(G.nodes()))
        # print('done generating graphs')

        data = {}
        data["size"] = len(G.nodes())
        data["density"] = nx.density(G)
        # data['density of largest component']=nx.density(G0)
        data["highest degree"] = max(degree_distribution(G).values())
        # data['highest degree of largest component']=max(degree_distribution(G0).values())
        data["average degree"] = average_degree(G)
        # data['average degree of largest component']=average_degree(G0)
        data["average clustering coefficient"] = nx.average_clustering(G)
        # data['average clustering coefficient of largest component']=nx.average_clustering(G0)
        ss_output = calculate_gate_ss(G)
        data["gates SS"] = ss_output[0] / n
        data["number of stars"] = len(ss_output[1])
        # print('done ss')
        data["gates MMG"] = calculate_gate_steiner(G, list(G.nodes())) / n
        # print('done MMG')
        data["gates MMG upd"] = calculate_gate_ss(steiner_G0)[0] / n
        # print('done MMG upd')

        return data

    def take_sample_subset(n, f, p):
        # print(0)
        G = gnp_random_connected_graph(n, p)
        m = int(np.ceil(n * f))
        S = generate_selected_nodes(G, m)
        G0 = generate_connected_subgraph_sid(G, S)
        steiner_G0 = generate_steiner_subgraph(G, S)

        data = {}
        data["size"] = len(G0.nodes())
        data["density"] = nx.density(G)
        data["density of subset"] = nx.density(G0)
        data["highest degree"] = max(degree_distribution(G).values())
        data["highest degree of subset"] = max(degree_distribution(G0).values())
        data["average degree"] = average_degree(G)
        data["average degree of subset"] = average_degree(G0)
        data["average clustering coefficient"] = nx.average_clustering(G)
        data["average clustering coefficient of subset"] = nx.average_clustering(G0)
        ss_output = calculate_gate_ss(G0)
        data["gates SS"] = ss_output[0] / m
        data["number of stars"] = len(ss_output[1])
        # print('done ss')
        data["gates MMG"] = calculate_gate_steiner(G0, list(G0.nodes())) / m
        # print('done MMG')
        data["gates MMG upd"] = calculate_gate_ss(steiner_G0)[0] / m
        # print('done MMG upd')

        return data

    output = {}

    for n in N:
        for p in P:
            for f in F:
                if display:
                    print(n, f, p)
                if f == 1:
                    output[n, f, p] = Parallel(n_jobs=8)(
                        delayed(take_sample)(n, p) for i in range(num_samples)
                    )
                    # output[n,d] = [take_sample(n,d) for i in range(num_samples)]

                    C[n, f, p].append(
                        np.mean([data["size"] for data in output[n, f, p]])
                    )
                    R[n, f, p].append(
                        np.mean([data["density"] for data in output[n, f, p]])
                    )
                    # R0[n,f].append(np.mean([data['density of subset'] for data in output[n,f,p]]))
                    K[n, f, p].append(
                        np.mean([data["highest degree"] for data in output[n, f, p]])
                    )
                    # K0[n,f].append(np.mean([data['highest degree of subset'] for data in output[n,f,p]]))
                    Cl[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    # Cl0[n,f].append(np.mean([data['average clustering coefficient of subset'] for data in output[n,f,p]]))
                    deg[n, f, p].append(
                        np.mean([data["average degree"] for data in output[n, f, p]])
                    )
                    # deg0[n,f].append(np.mean([data['average degree of subset'] for data in output[n,f,p]]))
                    gates_SS[n, f, p].append(
                        np.mean([data["gates SS"] for data in output[n, f, p]])
                    )
                    stars_SS[n, f, p].append(
                        np.mean([data["number of stars"] for data in output[n, f, p]])
                    )
                    gates_MMG[n, f, p].append(
                        np.mean([data["gates MMG"] for data in output[n, f, p]])
                    )
                    gates_MMGupd[n, f, p].append(
                        np.mean([data["gates MMG upd"] for data in output[n, f, p]])
                    )
                else:
                    output[n, f, p] = Parallel(n_jobs=8)(
                        delayed(take_sample_subset)(n, f, p) for i in range(num_samples)
                    )
                    # output[n,d] = [take_sample(n,d) for i in range(num_samples)]

                    C[n, f, p].append(
                        np.mean([data["size"] for data in output[n, f, p]])
                    )
                    R[n, f, p].append(
                        np.mean([data["density"] for data in output[n, f, p]])
                    )
                    R0[n, f, p].append(
                        np.mean([data["density of subset"] for data in output[n, f, p]])
                    )
                    K[n, f, p].append(
                        np.mean([data["highest degree"] for data in output[n, f, p]])
                    )
                    K0[n, f, p].append(
                        np.mean(
                            [
                                data["highest degree of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    Cl[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    Cl0[n, f, p].append(
                        np.mean(
                            [
                                data["average clustering coefficient of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    deg[n, f, p].append(
                        np.mean([data["average degree"] for data in output[n, f, p]])
                    )
                    deg0[n, f, p].append(
                        np.mean(
                            [
                                data["average degree of subset"]
                                for data in output[n, f, p]
                            ]
                        )
                    )
                    gates_SS[n, f, p].append(
                        np.mean([data["gates SS"] for data in output[n, f, p]])
                    )
                    stars_SS[n, f, p].append(
                        np.mean([data["number of stars"] for data in output[n, f, p]])
                    )
                    gates_MMG[n, f, p].append(
                        np.mean([data["gates MMG"] for data in output[n, f, p]])
                    )
                    gates_MMGupd[n, f, p].append(
                        np.mean([data["gates MMG upd"] for data in output[n, f, p]])
                    )

    output_avg = {}
    for n in N:
        for f in F:
            for p in P:
                output_avg[n, f, p] = {}
                output_avg[n, f, p]["size"] = np.array(C[n, f, p])
                output_avg[n, f, p]["density"] = np.array(R[n, f, p])
                output_avg[n, f, p]["density of subset"] = np.array(R0[n, f, p])
                output_avg[n, f, p]["highest degree"] = np.array(K[n, f, p])
                output_avg[n, f, p]["highest degree of subset"] = np.array(K0[n, f, p])
                output_avg[n, f, p]["average clustering coefficient"] = np.array(
                    Cl[n, f, p]
                )
                output_avg[n, f, p]["average clustering coefficient of subset"] = (
                    np.array(Cl0[n, f, p])
                )
                output_avg[n, f, p]["average degree"] = np.array(deg[n, f, p])
                output_avg[n, f, p]["average degree of subset"] = np.array(
                    deg0[n, f, p]
                )
                output_avg[n, f, p]["gates SS"] = np.array(gates_SS[n, f, p])
                output_avg[n, f, p]["number of stars"] = np.array(stars_SS[n, f, p])
                output_avg[n, f, p]["gates MMG"] = np.array(gates_MMG[n, f, p])
                output_avg[n, f, p]["gates MMG upd"] = np.array(gates_MMGupd[n, f, p])

    if save_to_file:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "data_ER_" + timestr + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump([N, F, P, num_samples, output_avg, output], f)

        return N, F, P, num_samples, output_avg, output

    else:
        return N, F, P, num_samples, output_avg, output


def generate_data_photonic(
    num_samples=500, circle=True, display=True, save_to_file=True
):

    # N=np.linspace(50,500,10,dtype=int)
    # N=np.linspace(50,200,1,dtype=int)
    N = [100, 200, 300, 400, 500]
    D = np.linspace(50, 1000, 20)  ### maximum distances
    # D=[800]

    print(
        "Generating data for photonic networks with number of nodes N =",
        N,
        "and distances D =",
        D,
    )
    print("\nThe number of samples is", num_samples)

    C = {}  ### size of the largest connected component
    R = {}  ### density of the full graph
    R0 = {}  ### density of the largest connected component
    K = {}  ### max degree of the full graph
    K0 = {}  ### max degree of the largest connected component
    Cl = {}  ### average clustering coefficient of the full graph
    Cl0 = {}  ### average clustering coefficient of the largest connected component
    deg = {}  ### average degree of the full graph
    deg0 = {}  ### average degree of the largest connected component
    gates_SS = {}
    gates_SS_normalized = {}
    stars_SS = {}
    gates_MMG = {}
    gates_MMG_normalized = {}
    gates_MMGupd = {}
    gates_MMGupd_normalized = {}

    for n in N:
        for d in D:
            C[n, d] = []
            R[n, d] = []
            R0[n, d] = []
            K[n, d] = []
            K0[n, d] = []
            Cl[n, d] = []
            Cl0[n, d] = []
            deg[n, d] = []
            deg0[n, d] = []
            gates_SS[n, d] = []
            gates_SS_normalized[n, d] = []
            stars_SS[n, d] = []
            gates_MMG[n, d] = []
            gates_MMG_normalized[n, d] = []
            gates_MMGupd[n, d] = []
            gates_MMGupd_normalized[n, d] = []

    def take_sample(n, d):
        # print(0)
        if circle:
            G = generate_random_geometric_graph_circle(n, d)
        else:
            G = generate_random_geometric_graph(n, d)
        S = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(S[0])  ### The largest connected component
        steiner_G0 = generate_steiner_subgraph(G0, list(G0.nodes()))
        # print('done generating graphs')

        data = {}
        data["size"] = len(G0.nodes())
        data["density"] = nx.density(G)
        data["density of largest component"] = nx.density(G0)
        data["highest degree"] = max(degree_distribution(G).values())
        data["highest degree of largest component"] = max(
            degree_distribution(G0).values()
        )
        data["average degree"] = average_degree(G)
        data["average degree of largest component"] = average_degree(G0)
        data["average clustering coefficient"] = nx.average_clustering(G)
        data["average clustering coefficient of largest component"] = (
            nx.average_clustering(G0)
        )
        output_SS = calculate_gate_ss(G0)
        data["gates SS"] = (output_SS[0], output_SS[0] / len(G0.nodes()))
        data["number of stars"] = len(output_SS[1])
        # print('done ss')
        output_steiner = calculate_gate_steiner(G0, list(G0.nodes()))
        data["gates MMG"] = (output_steiner, output_steiner / len(G0.nodes()))
        # print('done MMG')
        output_SS_steiner = calculate_gate_ss(steiner_G0)[0]
        data["gates MMG upd"] = (
            output_SS_steiner,
            output_SS_steiner / len(steiner_G0.nodes()),
        )
        # print('done MMG upd')

        return data

    output = {}

    for n in N:
        for d in D:
            if display:
                print(n, d)

            output[n, d] = Parallel(n_jobs=8)(
                delayed(take_sample)(n, d) for i in range(num_samples)
            )
            # output[n,d] = [take_sample(n,d) for i in range(num_samples)]

            C[n, d].append(np.mean([data["size"] for data in output[n, d]]))
            R[n, d].append(np.mean([data["density"] for data in output[n, d]]))
            R0[n, d].append(
                np.mean([data["density of largest component"] for data in output[n, d]])
            )
            K[n, d].append(np.mean([data["highest degree"] for data in output[n, d]]))
            K0[n, d].append(
                np.mean(
                    [
                        data["highest degree of largest component"]
                        for data in output[n, d]
                    ]
                )
            )
            Cl[n, d].append(
                np.mean(
                    [data["average clustering coefficient"] for data in output[n, d]]
                )
            )
            Cl0[n, d].append(
                np.mean(
                    [
                        data["average clustering coefficient of largest component"]
                        for data in output[n, d]
                    ]
                )
            )
            deg[n, d].append(np.mean([data["average degree"] for data in output[n, d]]))
            deg0[n, d].append(
                np.mean(
                    [
                        data["average degree of largest component"]
                        for data in output[n, d]
                    ]
                )
            )
            gates_SS[n, d].append(
                np.mean([data["gates SS"][0] for data in output[n, d]])
            )
            gates_SS_normalized[n, d].append(
                np.mean([data["gates SS"][1] for data in output[n, d]])
            )
            stars_SS[n, d].append(
                np.mean([data["number of stars"] for data in output[n, d]])
            )
            gates_MMG[n, d].append(
                np.mean([data["gates MMG"][0] for data in output[n, d]])
            )
            gates_MMG_normalized[n, d].append(
                np.mean([data["gates MMG"][1] for data in output[n, d]])
            )
            gates_MMGupd[n, d].append(
                np.mean([data["gates MMG upd"][0] for data in output[n, d]])
            )
            gates_MMGupd_normalized[n, d].append(
                np.mean([data["gates MMG upd"][1] for data in output[n, d]])
            )

    output_avg = {}
    for n in N:
        output_avg[n, d] = {}
        output_avg[n, d]["size"] = np.array(C[n, d])
        output_avg[n, d]["density"] = np.array(R[n, d])
        output_avg[n, d]["density of largest component"] = np.array(R0[n, d])
        output_avg[n, d]["highest degree"] = np.array(K[n, d])
        output_avg[n, d]["highest degree of largest component"] = np.array(K0[n, d])
        output_avg[n, d]["average clustering coefficient"] = np.array(Cl[n, d])
        output_avg[n, d]["average clustering coefficient of largest component"] = (
            np.array(Cl0[n, d])
        )
        output_avg[n, d]["average degree"] = np.array(deg[n, d])
        output_avg[n, d]["average degree of largest component"] = np.array(deg0[n, d])
        output_avg[n, d]["gates SS"] = np.array(gates_SS[n, d])
        output_avg[n, d]["gates SS normalized"] = np.array(gates_SS_normalized[n, d])
        output_avg[n, d]["number of stars"] = np.array(stars_SS[n, d])
        output_avg[n, d]["gates MMG"] = np.array(gates_MMG[n, d])
        output_avg[n, d]["gates MMG normalized"] = np.array(gates_MMG_normalized[n, d])
        output_avg[n, d]["gates MMG upd"] = np.array(gates_MMGupd[n, d])
        output_avg[n, d]["gates MMG upd normalized"] = np.array(
            gates_MMGupd_normalized[n, d]
        )

    if save_to_file:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "data_photonic_" + timestr + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump([N, D, num_samples, output_avg, output], f)

        return N, D, num_samples, output_avg, output

    else:
        return N, D, num_samples, output_avg, output


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


def generate_steiner_subgraph(G, selected_nodes):
    # Find the Steiner tree among the selected nodes
    # steiner_subgraph = nx.algorithms.approximation.steinertree.steiner_tree(G, selected_nodes)
    steiner_subgraph = nx.algorithms.approximation.steiner_tree(
        G, selected_nodes, method="mehlhorn"
    )
    return steiner_subgraph


def calculate_gate_steiner(G, selected_nodes):
    """
    Calculates the gates using the Steiner method (MMG method) on the given graph G.
    """

    if not G.edges():
        return 0
    else:

        steiner_subgraph = generate_steiner_subgraph(G, selected_nodes)

        saved_steiner_subgraph = steiner_subgraph
        visited_list = []
        non_visited_list = []
        unwanted_nodes = list(set(steiner_subgraph.nodes()) - set(selected_nodes))
        MMG19_gates = 0
        MMG19_gates_upd = 0
        temp_tree = steiner_subgraph.copy()
        degree_one_nodes = [
            node for node in temp_tree.nodes() if temp_tree.degree(node) == 1
        ]
        star_center_node = random.choice(degree_one_nodes)
        star_center_neighbors = list(temp_tree.neighbors(star_center_node))
        non_visited_list.extend(star_center_neighbors)
        visited_list.append(star_center_node)

        while len(non_visited_list) > 0:

            picked_node = non_visited_list.pop(0)

            k = temp_tree.degree([picked_node])[picked_node]

            if k == 1:
                visited_list.append(picked_node)
            else:
                if picked_node in unwanted_nodes:
                    value = (k * (k - 1) / 2) + k
                    value_upd = k - 1

                    MMG19_gates += value
                    MMG19_gates_upd += value_upd

                    for neighbor in temp_tree.neighbors(picked_node):
                        if neighbor != star_center_node:
                            temp_tree.add_edge(neighbor, star_center_node)

                    visited_list.append(picked_node)

                    star_center_neighbors = temp_tree.neighbors(star_center_node)
                    for neighbor in star_center_neighbors:
                        if (
                            neighbor not in visited_list
                            and neighbor not in non_visited_list
                        ):
                            non_visited_list.append(neighbor)

                    temp_tree.remove_node(picked_node)

                else:
                    value = (k * (k - 1) / 2) + (2 * k) - 1
                    value_upd = k - 1

                    # Add the calculated value to the MMG19_gates variable
                    MMG19_gates += value
                    MMG19_gates_upd += value_upd

                    for neighbor in temp_tree.neighbors(picked_node):
                        if neighbor != star_center_node:
                            temp_tree.add_edge(neighbor, star_center_node)

                    edges_to_remove = [
                        edge
                        for edge in temp_tree.edges(picked_node)
                        if edge[1] != star_center_node
                    ]
                    temp_tree.remove_edges_from(edges_to_remove)

                    visited_list.append(picked_node)

                    star_center_neighbors = temp_tree.neighbors(star_center_node)
                    for neighbor in star_center_neighbors:
                        if (
                            neighbor not in visited_list
                            and neighbor not in non_visited_list
                        ):
                            non_visited_list.append(neighbor)
        return MMG19_gates


def distance(G, e):
    """
    Distance between the nodes in the edge e of the graph G.
    """

    P = nx.get_node_attributes(G, "pos")

    Pos0 = P[e[0]]
    Pos1 = P[e[1]]

    x0 = Pos0[0]
    x1 = Pos1[0]
    y0 = Pos0[1]
    y1 = Pos1[1]

    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def distance_alt(p0, p1):

    x0 = p0[0]
    x1 = p1[0]
    y0 = p0[1]
    y1 = p1[1]

    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def edge_prob(d, L0=22):

    return np.exp(-d / L0)


def generate_random_geometric_graph(N, D, L0=22):
    """
    N is the number of nodes, D is the size of the box.
    (i.e., we place N nodes randomly in a box of size D x D.)
    """

    # D=1000  ### Total grid size
    # r=R/D ### radius
    # rho=0.001
    # N=int(rho*D**2)

    ### This generates N nodes within the unit square
    G = nx.random_geometric_graph(N, 0)  ### The graph contains only nodes, no edges

    G.graph = {"dist_param": D}

    P = nx.get_node_attributes(G, "pos")
    for v in list(P.keys()):
        pos = P[v]
        nx.set_node_attributes(G, dict([(v, [D * pos[0], D * pos[1]])]), "pos")

    F = list(combinations(G.nodes(), 2))
    # E=[]

    for f in F:
        d = distance(G, f)  # *D
        p = edge_prob(d, L0)
        if np.random.rand() < p:
            # E.append(f)
            G.add_edges_from([f])
            nx.set_edge_attributes(G, dict([(f, p)]), "prob")
            nx.set_edge_attributes(G, dict([(f, d)]), "dist")

    return G


def generate_random_positions_circle(N, R, as_dict=True):
    """
    N is the number of points, R is the radius, between 0 and 1.
    """

    theta = np.random.uniform(0, 2 * np.pi, N)
    radius = np.sqrt(np.random.uniform(0, R**2, N))

    y = radius * np.cos(theta) + np.sqrt(R**2)
    x = radius * np.sin(theta) + np.sqrt(R**2)

    if as_dict:
        return dict(zip(range(N), list(zip(x, y))))
    else:
        return x, y


def generate_random_geometric_graph_circle(N, D, L0=22):
    """
    N is the number of nodes, D is the diameter of the circle
    (i.e., maximum possible distance between two nodes)
    """

    G = nx.empty_graph(N)
    pos = generate_random_positions_circle(N, D / 2)
    nx.set_node_attributes(G, pos, "pos")

    G.graph["dist_param"] = D

    def dist(u, v):
        return distance_alt(pos[u], pos[v])

    def should_join(pair):
        return np.random.rand() < edge_prob(dist(*pair), L0)

    G.add_edges_from(filter(should_join, combinations(G, 2)))

    return G


def waxman_graph_circle(N, D, L0=22, L=None, beta=0.4, alpha=0.1, metric=None):
    """
    N is the number of nodes, D is the diameter of the circle
    (i.e., maximum possible distance between two nodes)
    """

    G = nx.empty_graph(N)
    pos = generate_random_positions_circle(N, D / 2)
    nx.set_node_attributes(G, pos, "pos")

    G.graph["dist_param"] = D

    if metric is None:
        metric = math.dist

    if L is None:
        L = max(metric(x, y) for x, y in combinations(pos.values(), 2))

        def dist(u, v):
            return metric(pos[u], pos[v])

    else:

        def dist(u, v):
            return metric(pos[u], pos[v])

        # def dist(u, v):
        #    return np.random.rand() * L

    # `pair` is the pair of nodes to decide whether to join.
    def should_join(pair):
        return np.random.rand() < math.exp(-dist(*pair) / L0) * beta * math.exp(
            -dist(*pair) / (alpha * L)
        )

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G


"""
OLD CODE 
def generate_random_geometric_graph_circle(N,D,L0=22):

    '''
    N is the number of nodes, D is the diameter (i.e., the
    maximum distance between any two nodes is D).
    '''

    G=nx.Graph()
    G.graph['dist_param']=D

    P=generate_random_positions_circle(N,D/2)

    G.add_nodes_from(list(P.keys()))
    nx.set_node_attributes(G,P,'pos')

    F=list(combinations(G.nodes(),2))
    #E=[]
    '''
    def add_edge(f):
        d=distance(G,f)#*D
        p=edge_prob(d,L0)
        if np.random.rand()<p:
            #E.append(f)
            G.add_edges_from([f])
            nx.set_edge_attributes(G,dict([(f,p)]),'prob')
            nx.set_edge_attributes(G,dict([(f,d)]),'dist')
        
        return f,p,d
    output=Parallel(n_jobs=8)(delayed(add_edge)(f) for f in F)
    '''
    
    for f in F:
        #d=distance(G,f)
        d=distance_alt(P[f[0]],P[f[1]])
        p=edge_prob(d,L0)
        if np.random.rand()<p:
            #E.append(f)
            G.add_edges_from([f])
            nx.set_edge_attributes(G,dict([(f,p)]),'prob')
            nx.set_edge_attributes(G,dict([(f,d)]),'dist')
            
    return G """


def draw_geometric_graph(G, D=None):

    plt.figure(figsize=(5, 5))

    Pos = nx.get_node_attributes(G, "pos")

    if D is None:
        D = G.graph["dist_param"]

    E = list(G.edges())

    for coord in Pos.values():
        plt.plot(coord[0], coord[1], "ob")

    for e in E:
        v0 = e[0]
        v1 = e[1]
        plt.plot([Pos[v0][0], Pos[v1][0]], [Pos[v0][1], Pos[v1][1]], "-k")

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim([-0.05 * D, D + 0.05 * D])
    plt.ylim([-0.05 * D, D + 0.05 * D])

    plt.show()


def degree_distribution(G, V=1, descending=True):
    """
    Gets the degree distribution of the graph G. The variable
    V is a list containing a subset of nodes.
    """

    # D_sorted=sorted(G.degree(),key=lambda x: x[1],reverse=True)
    # return dict(((n,d) for n, d in G.degree()))

    if V == 1:
        return dict(
            sorted(G.degree, key=lambda x: x[1], reverse=descending)
        )  ### returns the degrees in descending order
    else:
        return dict(
            sorted(
                [(v, G.degree(v)) for v in V], key=lambda x: x[1], reverse=descending
            )
        )


def average_degree(G):

    # dist=degree_distribution(G)

    # return np.mean(list(dist.values())),
    return 2 * len(G.edges()) / len(G.nodes())


def largest_connected_component(G):

    S = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(S[0])  ### The largest connected component

    return G0


def generate_selected_nodes(G, num_selected_nodes):
    # Generate a random list of selected nodes
    selected_nodes = random.sample(list(G.nodes()), num_selected_nodes)
    return selected_nodes


def generate_connected_subgraph_sid(G, selected_nodes):
    # Check connectivity and group the nodes
    node_groups = list(nx.connected_components(G.subgraph(selected_nodes)))

    # Store the groups in a list
    grouped_nodes = [list(group) for group in node_groups]
    grouped_nodes.sort(key=lambda x: -len(x))

    # Check if there's only one group in grouped_nodes
    if len(grouped_nodes) == 1:
        connected_nodes = grouped_nodes[0]

        # Create a subgraph with connected nodes
        connected_subgraph = G.subgraph(connected_nodes)
        return connected_subgraph

    else:
        # Create a list named 'connected_nodes' and put the nodes from the largest connected component in it
        random_node = random.choice(grouped_nodes[0])
        connected_nodes = [node for node in grouped_nodes[0]]
        grouped_nodes.pop(0)

        # Create a BFS traversal and store nodes at each level
        bfs_layers = defaultdict(list)
        visited = set()  # To keep track of visited nodes

        # Define the root node as the random_node
        root_node = random_node

        # Initialize the BFS queue with the root node
        queue = deque([(root_node, 0)])  # The second element of the tuple is the level

        while queue:
            node, level = queue.popleft()

            # Check if the node has already been visited at any previous level
            if node not in visited:
                bfs_layers[level].append(node)
                visited.add(node)

                # Enqueue unvisited neighbors at the next level
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, level + 1))

        # Randomly choose a node from the BFS traversal
        random_node = random.choice(list(visited))

        # Find the level of the randomly chosen node
        level = -1
        for l, nodes in bfs_layers.items():
            if random_node in nodes:
                level = l
                break

        # Choose a node from the first element of grouped_nodes
        while grouped_nodes:
            first_group = grouped_nodes[0]
            random_node = random.choice(first_group)

            # Find the level of the randomly chosen node
            level = -1
            for l, nodes in bfs_layers.items():
                if random_node in nodes:
                    level = l
                    break

            # Initialize a list to store nodes from the randomly chosen node to the root
            nodes_from_random_to_root = [random_node]

            # Trace back to the root node, collecting all nodes with only one neighbor at each level
            current_node = random_node
            while current_node != root_node:
                level -= 1
                next_node = None
                for neighbor in G.neighbors(current_node):
                    if neighbor in bfs_layers[level]:
                        next_node = neighbor
                        break
                if next_node is not None:
                    nodes_from_random_to_root.append(next_node)
                    current_node = next_node
                else:
                    # If there are no more valid neighbors at the current level, break out of the loop
                    break

            # Add nodes_from_random_to_root to connected_nodes without repeating
            connected_nodes.extend(
                node
                for node in nodes_from_random_to_root
                if node not in connected_nodes
            )

            # Add all nodes in the first_group to connected_nodes without repeating
            connected_nodes.extend(
                node for node in first_group if node not in connected_nodes
            )

            # Remove the first element from grouped_nodes
            grouped_nodes.pop(0)

        # Create a subgraph with connected nodes
        connected_subgraph = G.subgraph(connected_nodes)
        return connected_subgraph
