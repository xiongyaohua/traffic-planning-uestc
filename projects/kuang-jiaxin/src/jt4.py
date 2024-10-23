# -*- coding: gbk -*-
# 设定文件编码为GBK，以支持中文字符。

import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# 定义网络中的节点（减少两个节点，去掉'G'和'H'）
node_list = ['A', 'B', 'C', 'D', 'E', 'F']

# 初始化网络图，采用字典结构存储节点及其邻居关系和边的权重
net_graph = {node: {} for node in node_list}

# 全局变量，用于在更新流量时调整权重
flow_factor = 1000


# 使用Dijkstra算法计算从起点到终点的最短路径
def find_shortest_path(graph, start_node, target_node):
    priority_queue = [(0, start_node, [])]  # 优先队列，存储节点的累计权重、节点及路径
    visited_nodes = set()  # 存储已经访问过的节点
    min_distance = {start_node: 0}  # 记录从起点到其他节点的最小距离

    while priority_queue:
        # 取出当前权重最小的节点
        current_cost, current_node, current_path = heapq.heappop(priority_queue)

        if current_node in visited_nodes:
            continue  # 跳过已经访问的节点

        visited_nodes.add(current_node)
        current_path = current_path + [current_node]  # 更新路径

        if current_node == target_node:
            return current_cost, current_path  # 到达目标节点，返回总成本和路径

        # 遍历当前节点的邻接节点
        for neighbor_node, edge_weight in graph[current_node].items():
            if neighbor_node in visited_nodes:
                continue  # 跳过访问过的节点
            accumulated_distance = current_cost + edge_weight  # 累加边的权重
            if neighbor_node not in min_distance or accumulated_distance < min_distance[neighbor_node]:
                min_distance[neighbor_node] = accumulated_distance
                heapq.heappush(priority_queue, (accumulated_distance, neighbor_node, current_path))

    return float("inf"), []  # 如果无法到达目标，返回无穷大和空路径


# 根据路径更新流量分布
def adjust_flow_distribution(q_map, path_nodes):
    # 衰减所有边的流量
    for node, neighbors in net_graph.items():
        for neighbor, flow_value in neighbors.items():
            q_map[node][neighbor] *= 0.9  # 流量减少10%

    # 对路径上的边进行流量增加
    for idx in range(len(path_nodes) - 1):
        current_node = path_nodes[idx]
        next_node = path_nodes[idx + 1]
        if next_node in q_map[current_node]:
            q_map[current_node][next_node] += flow_factor * 0.1  # 增加流量10%


# 更新网络中边的权重
def update_weights(graph, q_map):
    for node, neighbors in graph.items():
        for neighbor, edge_weight in neighbors.items():
            graph[node][neighbor] *= transmission_function(q_map[node][neighbor])  # 使用传输函数更新边的权重


# 定义传输函数，根据流量q的大小调整边的权重
def transmission_function(q_value):
    return 1 + (q_value / 1000) ** 2  # q值的平方决定边的权重增长


# 使用networkx库对网络进行可视化
def display_network(graph, layout_pos, axis):
    netx_graph = nx.DiGraph()  # 创建有向图

    # 添加节点和边
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            netx_graph.add_edge(node, neighbor, weight=weight)

    edge_data = netx_graph.edges(data=True)
    edge_thickness = [data['weight'] / 50 for (src, dst, data) in edge_data]  # 根据权重调整边的宽度

    axis.clear()  # 清除先前的图像
    nx.draw(netx_graph, layout_pos, with_labels=True, node_color='lightgreen', node_size=600, font_size=9, ax=axis)
    nx.draw_networkx_edge_labels(netx_graph, layout_pos, edge_labels={(src, dst): data['weight'] for (src, dst, data) in edge_data}, ax=axis)
    nx.draw_networkx_edges(netx_graph, layout_pos, edgelist=edge_data, width=edge_thickness, ax=axis)


# 图的动态迭代显示过程
def network_simulation(graph, q_map, flow_factor):
    figure, axis = plt.subplots()  # 创建画布

    # 使用有向图创建初始网络
    netx_graph = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            netx_graph.add_edge(node, neighbor, weight=weight)

    # 设置布局
    layout_pos = nx.shell_layout(netx_graph)

    # 定义每帧的更新函数
    def update_frame(frame_idx):
        shortest_dist, path = find_shortest_path(graph, 'A', 'F')
        adjust_flow_distribution(q_map, path)
        update_weights(graph, q_map)
        display_network(q_map, layout_pos, axis)
        axis.set_title(f"Iteration {frame_idx + 1}")

    # 动态绘制
    animation = FuncAnimation(figure, update_frame, frames=100, repeat=False)
    plt.show()


# 为节点生成随机边及其权重
for i in range(len(node_list)):
    edge_count = 0  # 记录节点的边数量
    for j in range(i + 1, len(node_list)):
        if random.random() > 0.3 and edge_count < 4:
            net_graph[node_list[i]][node_list[j]] = random.randint(1, 10)
            edge_count += 1

# 计算最短路径
shortest_distance, shortest_path = find_shortest_path(net_graph, 'A', 'F')

# 初始化q图，记录路径流量
q_map = {node: {} for node in node_list}
for node in net_graph:
    for neighbor in net_graph[node]:
        if node in shortest_path and neighbor in shortest_path and shortest_path.index(neighbor) == shortest_path.index(node) + 1:
            q_map[node][neighbor] = 1000  # 初始路径流量
        else:
            q_map[node][neighbor] = 0  # 其他路径初始流量为0

# 进行迭代模拟
network_simulation(net_graph, q_map, flow_factor)
