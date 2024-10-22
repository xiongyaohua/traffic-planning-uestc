import matplotlib.pyplot as plt
import networkx as nx

Network = dict

network = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": [""],
}

def add_node(network, node):
    network[node] = []

def add_link(network, node1, node2):
    network[node1].append(node2)

network = {}

add_node(network, "A")
add_node(network, "B")
add_node(network, "C")
add_link(network, "A", "B")
add_link(network, "A", "C")
add_link(network, "B", "C")

print(network)

AttributeMap = dict

length_map = {
    ("A", "B"): 4,
    ("A", "C"): 3,
    ("B", "C"): 7,
}

speed_map = {
    ("A", "B"): 40,
    ("A", "C"): 40,
    ("B", "C"): 60,
}

Path = list

def dijkstra(network, attr_map, origin, destination) -> Path:
    import heapq
    queue = [(0, origin, [])]
    seen = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in seen:
            continue
        path = path + [node]
        if node == destination:
            return path
        seen.add(node)
        for neighbor in network.get(node, []):
            if neighbor:
                heapq.heappush(queue, (cost + attr_map.get((node, neighbor), float('inf')), neighbor, path))
    return []

def assign(network, cost_function_map, od_pairs) -> AttributeMap:
    flow_map = {}
    for od_pair in od_pairs:
        new_flow_map = single_od_assign(network, cost_function_map, od_pair, flow_map)
        flow_map = flow_map_combine(flow_map, new_flow_map)
    return flow_map

def single_od_assign(network, cost_function_map, od_pair, base_flow_map) -> AttributeMap:
    origin, destination, flow = od_pair
    cost_map = update_cost(cost_function_map, base_flow_map)
    shortest_path = dijkstra(network, cost_map, origin, destination)
    flow_map = flow_loading(network, shortest_path, flow)

    for i in range(100):
        cost_map = update_cost(cost_function_map, flow_map_combine(base_flow_map, flow_map))
        shortest_path = dijkstra(network, cost_map, origin, destination)

        flow_map = flow_map_scale(flow_map, 0.9)
        new_flow_map = flow_loading(network, shortest_path, flow * 0.1)
        flow_map = flow_map_combine(flow_map, new_flow_map)
    return flow_map

def flow_map_combine(map1, map2):
    combined_map = map1.copy()
    for key in map2:
        if key in combined_map:
            combined_map[key] += map2[key]
        else:
            combined_map[key] = map2[key]
    return combined_map

def flow_map_scale(map1, scale):
    return {key: value * scale for key, value in map1.items()}

def update_cost(cost_function_map, flow_map):
    cost_map = {}
    for key in cost_function_map:
        cost_function = cost_function_map[key]
        flow = flow_map.get(key, 0)
        cost = cost_function(flow)
        cost_map[key] = cost
    return cost_map

def flow_loading(network, path, flow):
    flow_map = {}
    links = set(zip(path[:-1], path[1:]))
    for link in network_links(network):
        flow_map[link] = flow if link in links else 0
    return flow_map

def network_links(network):
    links = []
    for node, neighbors in network.items():
        for neighbor in neighbors:
            if neighbor:
                links.append((node, neighbor))
    return links

def draw_network(network, flow_map):
    G = nx.DiGraph()
    for node in network:
        G.add_node(node)
    for (node1, node2), flow in flow_map.items():
        G.add_edge(node1, node2, weight=flow)

    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in edges})
    nx.draw(G, pos, edges=edges, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)

    plt.show()

# od_pairs = [("A", "C", 10)]
# cost_function_map = {
#     ("A", "B"): lambda flow: 10 + flow,
#     ("A", "C"): lambda flow: 5 + 2 * flow,
#     ("B", "C"): lambda flow: 7 + 3 * flow,
# }

# flow_map = assign(network, cost_function_map, od_pairs)
# draw_network(network, flow_map)