from time import perf_counter
import networkx as nx
from cdlib.classes import NodeClustering
from cdlib.algorithms import louvain
from math import log
from utils import read_graph, generate_erdos
from random import sample, randrange
import os

def pref_attachment_index(graph: nx.Graph, i, j):
    return graph.degree[i] * graph.degree[j]

def adamic_adar_index(graph: nx.Graph, i, j):
    neighbors_i = set(list(graph[i]))
    neighbors_j = set(list(graph[j]))
    intersection = list(neighbors_i.intersection(neighbors_j))
    return sum([1/log(graph.degree[node]) for node in intersection])

def get_community(i, communities):
    for index, community in enumerate(communities):
        if i in community:
            return index

def louvain_index(graph: nx.Graph, i, j, louvain_clustering: NodeClustering):
    # First find communities of nodes i and j
    i_comm = get_community(i, louvain_clustering.communities)
    j_comm = get_community(j, louvain_clustering.communities)
    
    if i_comm != j_comm:
        return 0

    community = louvain_clustering.communities[i_comm]
    induced_graph = graph.subgraph(community)
    n_c = induced_graph.number_of_nodes() # or len(community)
    m_c = induced_graph.number_of_edges()

    return 2*m_c / (n_c*(n_c - 1))

def AUC(m1, m2, m):
    """
    m1 ~ m'
    m2 ~ m''
    """
    return (m1 + m2 / 2) / (m / 10)

def equal(float_1, float_2):
    return abs(float_1 - float_2) < 0.00001

def evaluate_link_prediction(method, graph: nx.Graph, louvain_clustering: NodeClustering):
    """
    method - A link prediction method that returns a community index for two nodes
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes)
    m = graph.number_of_edges()
    edges = list(graph.edges)
    sample_size = round(m / 10)
    Ln = set()
    Lp = set()

    while True:
        # using itertools.combinations is too costly for 25000 nodes so
        start_node = nodes[randrange(0, n)]
        break_out = False
        for non_neighbor in graph.nodes - set(graph[start_node]): # not linked to these
            Ln.add((start_node, non_neighbor))
            if len(Ln) == sample_size:
                break_out = True
                break
        if break_out:
            break
    print(f"\t\tLn created ({method.__name__})")
    
    while True:
        edge = edges[randrange(0, m)] # if we sample the same edge, it's fine
        Lp.add(edge)
        if len(Lp) == sample_size:
            break
    print(f"\t\tLp created ({method.__name__})")

    # Now to calculate AUC
    m1 = 0
    m2 = 0

    for i in range(sample_size):
        Lp_pair = sample(Lp, 1)[0]
        Ln_pair = sample(Ln, 1)[0]
        Lp_args = [graph, *Lp_pair]
        Ln_args = [graph, *Ln_pair]
        if louvain_clustering:
            Lp_args.append(louvain_clustering)
            Ln_args.append(louvain_clustering)

        Lp_index = method(*Lp_args)
        Ln_index = method(*Lp_args)
        if i % 1500 == 0:
            print(f"\t\tCompared {i}/{sample_size} indexes ({method.__name__})")
        if Lp_index > Ln_index:
            m1 += 1
        elif equal(Lp_index, Ln_index):
            m2 += 1

    return AUC(m1, m2, m)

def compute_average_auc(graph_name, link_prediction_methods, num_runs, remove_file):
    file_name = f"link_pred_{graph_name}.txt"
    file_name_time = f"link_pred_{graph_name}_time.txt"
    if remove_file and os.path.exists(file_name):
        os.remove(file_name)

    if graph_name == "erdos":
        graph, comms = generate_erdos(25000, 10)
    else:
        graph, comms = read_graph(graph_name)

    louvain_clustering = None
    if louvain_index in link_prediction_methods:
        print(f"Calculating Louvain clustering for {graph_name}")
        # even though Louvain is not entirely deterministic,
        # letting it run for 10 times on graphs with tens of thousands
        # of nodes would take an eternity, so I put it here
        # so it's only calculated once
        louvain_clustering = louvain(graph)
    print("Louvain clustering finished!")

    for method in link_prediction_methods:
        print(f"Trying out method {method.__name__} on {graph_name} graph")
        average_auc = 0
        start = perf_counter()
        louv = louvain_clustering if method.__name__ == "louvain_index" else None
        print(f"Louvain parameter: {louv}")
        for i in range(num_runs):
            print(f"\tIteration {i}/{num_runs}")
            average_auc += evaluate_link_prediction(method, graph, louv)
        
        time_taken = round(perf_counter() - start, 3)
        average_auc /= num_runs

        read_mode = "w" if not os.path.exists(file_name) else "a"
        with open(file_name, read_mode) as f:
            f.write(f"Method: {method.__name__} - {average_auc} average AUC\n")
        with open(file_name_time, "a") as f:
            f.write(f"Method: {method.__name__} - {time_taken} seconds for {num_runs} runs\n")

        print(f"Completed method {method.__name__} on {graph_name} graph!")
        print()

num_runs = 20
methods = [pref_attachment_index, adamic_adar_index, louvain_index]
# methods = [louvain_index]
for graph_name in ["circles", "erdos", "gnutella", "nec"]:
    compute_average_auc(graph_name, methods, num_runs, True)
"""
I tested the "louvain index" over 10 runs cause it takes a butt load of time
while the other two, preferential attachment and Adamic Abar index, i tested
over 20 runs
"""