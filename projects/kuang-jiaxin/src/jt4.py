# -*- coding: gbk -*-
# �趨�ļ�����ΪGBK����֧�������ַ���

import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ���������еĽڵ㣨���������ڵ㣬ȥ��'G'��'H'��
node_list = ['A', 'B', 'C', 'D', 'E', 'F']

# ��ʼ������ͼ�������ֵ�ṹ�洢�ڵ㼰���ھӹ�ϵ�ͱߵ�Ȩ��
net_graph = {node: {} for node in node_list}

# ȫ�ֱ����������ڸ�������ʱ����Ȩ��
flow_factor = 1000


# ʹ��Dijkstra�㷨�������㵽�յ�����·��
def find_shortest_path(graph, start_node, target_node):
    priority_queue = [(0, start_node, [])]  # ���ȶ��У��洢�ڵ���ۼ�Ȩ�ء��ڵ㼰·��
    visited_nodes = set()  # �洢�Ѿ����ʹ��Ľڵ�
    min_distance = {start_node: 0}  # ��¼����㵽�����ڵ����С����

    while priority_queue:
        # ȡ����ǰȨ����С�Ľڵ�
        current_cost, current_node, current_path = heapq.heappop(priority_queue)

        if current_node in visited_nodes:
            continue  # �����Ѿ����ʵĽڵ�

        visited_nodes.add(current_node)
        current_path = current_path + [current_node]  # ����·��

        if current_node == target_node:
            return current_cost, current_path  # ����Ŀ��ڵ㣬�����ܳɱ���·��

        # ������ǰ�ڵ���ڽӽڵ�
        for neighbor_node, edge_weight in graph[current_node].items():
            if neighbor_node in visited_nodes:
                continue  # �������ʹ��Ľڵ�
            accumulated_distance = current_cost + edge_weight  # �ۼӱߵ�Ȩ��
            if neighbor_node not in min_distance or accumulated_distance < min_distance[neighbor_node]:
                min_distance[neighbor_node] = accumulated_distance
                heapq.heappush(priority_queue, (accumulated_distance, neighbor_node, current_path))

    return float("inf"), []  # ����޷�����Ŀ�꣬���������Ϳ�·��


# ����·�����������ֲ�
def adjust_flow_distribution(q_map, path_nodes):
    # ˥�����бߵ�����
    for node, neighbors in net_graph.items():
        for neighbor, flow_value in neighbors.items():
            q_map[node][neighbor] *= 0.9  # ��������10%

    # ��·���ϵı߽�����������
    for idx in range(len(path_nodes) - 1):
        current_node = path_nodes[idx]
        next_node = path_nodes[idx + 1]
        if next_node in q_map[current_node]:
            q_map[current_node][next_node] += flow_factor * 0.1  # ��������10%


# ���������бߵ�Ȩ��
def update_weights(graph, q_map):
    for node, neighbors in graph.items():
        for neighbor, edge_weight in neighbors.items():
            graph[node][neighbor] *= transmission_function(q_map[node][neighbor])  # ʹ�ô��亯�����±ߵ�Ȩ��


# ���崫�亯������������q�Ĵ�С�����ߵ�Ȩ��
def transmission_function(q_value):
    return 1 + (q_value / 1000) ** 2  # qֵ��ƽ�������ߵ�Ȩ������


# ʹ��networkx���������п��ӻ�
def display_network(graph, layout_pos, axis):
    netx_graph = nx.DiGraph()  # ��������ͼ

    # ��ӽڵ�ͱ�
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            netx_graph.add_edge(node, neighbor, weight=weight)

    edge_data = netx_graph.edges(data=True)
    edge_thickness = [data['weight'] / 50 for (src, dst, data) in edge_data]  # ����Ȩ�ص����ߵĿ��

    axis.clear()  # �����ǰ��ͼ��
    nx.draw(netx_graph, layout_pos, with_labels=True, node_color='lightgreen', node_size=600, font_size=9, ax=axis)
    nx.draw_networkx_edge_labels(netx_graph, layout_pos, edge_labels={(src, dst): data['weight'] for (src, dst, data) in edge_data}, ax=axis)
    nx.draw_networkx_edges(netx_graph, layout_pos, edgelist=edge_data, width=edge_thickness, ax=axis)


# ͼ�Ķ�̬������ʾ����
def network_simulation(graph, q_map, flow_factor):
    figure, axis = plt.subplots()  # ��������

    # ʹ������ͼ������ʼ����
    netx_graph = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            netx_graph.add_edge(node, neighbor, weight=weight)

    # ���ò���
    layout_pos = nx.shell_layout(netx_graph)

    # ����ÿ֡�ĸ��º���
    def update_frame(frame_idx):
        shortest_dist, path = find_shortest_path(graph, 'A', 'F')
        adjust_flow_distribution(q_map, path)
        update_weights(graph, q_map)
        display_network(q_map, layout_pos, axis)
        axis.set_title(f"Iteration {frame_idx + 1}")

    # ��̬����
    animation = FuncAnimation(figure, update_frame, frames=100, repeat=False)
    plt.show()


# Ϊ�ڵ���������߼���Ȩ��
for i in range(len(node_list)):
    edge_count = 0  # ��¼�ڵ�ı�����
    for j in range(i + 1, len(node_list)):
        if random.random() > 0.3 and edge_count < 4:
            net_graph[node_list[i]][node_list[j]] = random.randint(1, 10)
            edge_count += 1

# �������·��
shortest_distance, shortest_path = find_shortest_path(net_graph, 'A', 'F')

# ��ʼ��qͼ����¼·������
q_map = {node: {} for node in node_list}
for node in net_graph:
    for neighbor in net_graph[node]:
        if node in shortest_path and neighbor in shortest_path and shortest_path.index(neighbor) == shortest_path.index(node) + 1:
            q_map[node][neighbor] = 1000  # ��ʼ·������
        else:
            q_map[node][neighbor] = 0  # ����·����ʼ����Ϊ0

# ���е���ģ��
network_simulation(net_graph, q_map, flow_factor)
