from random import random
import networkx as nx
from cdlib.classes import NodeClustering
from cdlib.algorithms import label_propagation, louvain
from infomap import Infomap
from cdlib.evaluation import normalized_mutual_information, variation_of_information
from math import log
import matplotlib.pyplot as plt
from time import perf_counter
import os
from utils import generate_erdos, read_graph

"""
1. Louvain
2. Infomap
3. Label propagation
"""

def infomap(graph: nx.Graph):
    """
    A wrapper function for the infomap algorithm
    """
    im = Infomap(silent=True)
    im.add_networkx_graph(graph)
    im.run()
    communities = {}
    for node in im.tree:
        if node.is_leaf:
            cluster = node.module_id
            node_number = node.node_id
            if cluster not in communities:
                communities[cluster] = []
            communities[cluster].append(node_number)
    
    return NodeClustering(list(communities.values()), graph, "infomap")

#########################
##### Girvan-Newman #####
#########################

def generate_girvan(mu):
    """
    Generate a Girvan Newman benchmark graph, with
    n = 72, 3 groups and average degree of 20
    """
    graph = nx.Graph(name = "girvan newman")
    n = 72
    cluster_size = 24
    avg_degree = 20
    communities = {}
    for i in range(n):
        cluster = i // cluster_size + 1
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(i)
        graph.add_node(i, cluster=cluster)

    # Add edges
    for i in range(n):
        for j in range(i + 1, n):
            if graph.nodes[i]['cluster'] == graph.nodes[j]['cluster']:
                # 20(1-mi) / 23 -> if they are the same cluster; 
                # 20 because that is its expected degree, 
                # 1-mi cause for mi = 0 we want all 20 links to be "inside"
                if random() < avg_degree * (1 - mu) / (cluster_size - 1):
                    graph.add_edge(i, j)
            else:
                # 20*mi / 48 -> if they are a different cluster, 
                # 48 cause that's 24 + 24 - number of nodes
                # in the other communities
                if random() < avg_degree * mu / (n - cluster_size):
                    graph.add_edge(i, j)

    return graph, NodeClustering(list(communities.values()), graph, "actual")

def girvan_newman(mus, algo):
    nmis = []
    avg_time = 0
    for mu in mus:
        nmi = 0
        for _ in range(25):
            graph, actual_comms = generate_girvan(mu)
            start = perf_counter()
            predicted_comms = algo(graph)
            avg_time += perf_counter() - start
            nmi += normalized_mutual_information(predicted_comms, actual_comms).score
        nmi /= 25
        nmis.append(nmi)
    avg_time /= 25 * len(mus)
    return nmis, avg_time

#########################
##### Lancichinetti #####
#########################

def lancichinetti(mus, algo):
    nmis = []
    avg_time = 0
    for mu in mus:
        m = int(mu * 10)
        nmi = 0
        for i in range(25):
            graphname = f"LFR_0{m}_{i}"
            print(f"\t\tAt graph {graphname} ({algo.__name__})")
            graph, actual_comms = read_graph(graphname)
            start = perf_counter()
            predicted_comms = algo(graph)
            avg_time += perf_counter() - start
            nmi += normalized_mutual_information(predicted_comms, actual_comms).score
        nmi /= 25
        nmis.append(nmi)
    avg_time /= 25 * len(mus)
    return nmis, avg_time

#######################
##### Erdos Renyi #####
#######################

def erdos_renyi(avg_degrees, algo):
    nvis = []
    normalization_constant = log(1000) # for normalized variation of information
    avg_time = 0
    for avg_degree in avg_degrees:
        nvi = 0
        for _ in range(25):
            graph, actual_comms = generate_erdos(1000, avg_degree)
            start = perf_counter()
            predicted_comms = algo(graph)
            avg_time += perf_counter() - start
            score = variation_of_information(predicted_comms, actual_comms).score
            nvi += score / normalization_constant
        nvi /= 25
        nvis.append(nvi)
    avg_time /= 25 * len(avg_degrees)
    return nvis, avg_time

#######################################
##### Lusseau bottlenose dolphins #####
#######################################

def dolfins(algo):
    graph, actual = read_graph("dolphins")
    n = graph.number_of_nodes()
    normalization_constant = log(n)
    all_clusterings = []
    for i in range(25):
        predicted_comms = algo(graph)
        all_clusterings.append(predicted_comms)
        
    # Calculate pair wise NVI and average the result
    l = len(all_clusterings)
    num_pairs = l*(l-1) // 2
    nvi = 0
    for i in range(l):
        clustering_1 = all_clusterings[i]
        for j in range(i+1, l):
            clustering_2 = all_clusterings[j]
            score = variation_of_information(clustering_1, clustering_2).score 
            nvi += score / normalization_constant
        # print(f"Iteration {i} done")
    return nvi / num_pairs

################
##### Plot #####
################

def plot(func, input_list, title, xlabel, metric):
    """
    General plotting function

    func: girvan_newman, erdos_renyi or lancichinetti
    input_list: list of mus, or list of average degrees for erdos_renyi
    algo_list: at most [louvain, infomap, label_propagation]
    title: title for the plot
    xlabel: what's on the x axis
    metric: which (average) metric was calculated (NVI for erdos_renyi, NMI for rest)
    """
    algo_list = [louvain, infomap, label_propagation]
    colors = ["firebrick", "forestgreen", "goldenrod"]
    file_name = f"comm_det_{func.__name__}.txt"
    if os.path.exists(file_name):
        os.remove(file_name)
    print(f"Function: {func.__name__}")
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f"Average {metric}")

    for i, algo in enumerate(algo_list):
        print(f"\tCurrent algorithm: {algo.__name__}")
        average_metric, average_time = func(input_list, algo)
        read_mode = "w" if not os.path.exists(file_name) else "a"
        with open(file_name, read_mode) as f:
            f.write(f"{algo.__name__} - {str(average_time)}\n")
        plt.plot(input_list, average_metric, label=f"{algo.__name__}", color=colors[i])
        plt.plot(input_list, average_metric, 'o', color="black")
    
    print()
    plt.xticks(input_list, input_list)
    plt.legend()
    plt.subplots_adjust(left=0.095, bottom=0.112, right=0.943, top=0.924)
    plt.show()

girvan_input = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
lancichinetti_input = [0, 0.2, 0.4, 0.6, 0.8]
erdos_input = [8, 16, 24, 32, 40]

"""Girvan Newman"""
plot(girvan_newman, girvan_input, "Girvan Newman accuracy", r"$\mu$", "NMI")

"""Lancichinetti"""
plot(lancichinetti, lancichinetti_input, "Lancichinetti accuracy", r"$\mu$", "NMI")

"""Erdos Renyi"""
plot(erdos_renyi, erdos_input, "Erdos Renyi robustness", "Average degree", "NVI")

"""Lusseau dolphin networks"""
# print("Function: dolfins")
# for algo in [louvain, infomap, label_propagation]:
#     print(f"\t{algo.__name__} - {dolfins(algo)} average pair-wise NVI")
    
"""
mu = 0 -> 0% of links are between the groups
mu = 0.2 -> 20% of links are between the groups

For mu = 0, NMI should equal 1, since finding the communities is
trivial, as they are disconnected components
We expect the performance to fall as mu increases

NMI - larger = better
NVI - smaller = better
"""
