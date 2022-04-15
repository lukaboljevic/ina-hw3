import networkx as nx
from cdlib.classes import NodeClustering

def generate_erdos(n, avg_degree):
    """
    Average degree of Erdos Renyi graph
    <k> = (n-1) * p => p = <k> / (n-1)
    """
    p = avg_degree / (n - 1)
    graph = nx.gnp_random_graph(n, p)
    communities = {}
    for i, c in enumerate(nx.connected_components(graph)):
        communities[i] = list(c)

    return graph, NodeClustering(list(communities.values()), graph, "actual")

def read_graph(graphname):
    """
    Read a .net file, and return the graph and the communities
    """
    G = nx.Graph(name=graphname)
    communities = {}
    with open(f"graphs/{graphname}.net", 'r', encoding='utf8') as f:
        f.readline()
        
        for line in f:
            if line.startswith("*"):
                break
            else:
                node_info = line.split("\"")
                node = int(node_info[0]) - 1
                label = node_info[1]
                cluster = int(node_info[2]) if len(node_info) > 2 and len(node_info[2].strip()) > 0 else None
                G.add_node(node, label=label, cluster=cluster)
                if cluster:
                    if cluster not in communities:
                        communities[cluster] = []
                    communities[cluster].append(node)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str) - 1, int(node2_str) - 1)

    return G, NodeClustering(list(communities.values()), G, "actual")

