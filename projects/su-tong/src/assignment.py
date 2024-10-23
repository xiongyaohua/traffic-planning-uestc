import heapq
import math
import time

import networkx as nx
import scipy

from network_import import *
from utils import PathUtils


class FlowTransportNetwork:

    def __init__(self):
        self.linkSet = {}
        self.nodeSet = {}

        self.tripSet = {}
        self.zoneSet = {}
        self.originZones = {}

        self.networkx_graph = None

    def to_networkx(self):
        if self.networkx_graph is None:
            self.networkx_graph = nx.DiGraph([(int(begin),int(end)) for (begin,end) in self.linkSet.keys()])
        return self.networkx_graph

    def reset_flow(self):
        for link in self.linkSet.values():
            link.reset_flow()

    def reset(self):
        for link in self.linkSet.values():
            link.reset()


class Zone:
    def __init__(self, zoneId: str):
        self.zoneId = zoneId

        self.lat = 0
        self.lon = 0
        self.destList = []  # 区域ID列表（字符串）


class Node:
    """
    此类包含与任何节点相关的属性
    """

    def __init__(self, nodeId: str):
        self.Id = nodeId

        self.lat = 0
        self.lon = 0

        self.outLinks = []  # 出节点ID列表（字符串）
        self.inLinks = []  # 入节点ID列表（字符串）

        # 用于Dijkstra算法
        self.label = np.inf
        self.pred = None


class Link:
    """
    此类包含与任何链接相关的属性
    """

    def __init__(self,
                 init_node: str,
                 term_node: str,
                 capacity: float,
                 length: float,
                 fft: float,
                 b: float,
                 power: float,
                 speed_limit: float,
                 toll: float,
                 linkType
                 ):
        self.init_node = init_node
        self.term_node = term_node
        self.max_capacity = float(capacity)  # 每小时车辆数
        self.length = float(length)  # 长度
        self.fft = float(fft)  # 自由流行驶时间（分钟）
        self.beta = float(power)
        self.alpha = float(b)
        self.speedLimit = float(speed_limit)
        self.toll = float(toll)
        self.linkType = linkType

        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.flow = 0.0
        self.cost = self.fft

    # 未用于作业的方法
    def modify_capacity(self, delta_percentage: float):
        assert -1 <= delta_percentage <= 1
        self.curr_capacity_percentage += delta_percentage
        self.curr_capacity_percentage = max(0, min(1, self.curr_capacity_percentage))
        self.capacity = self.max_capacity * self.curr_capacity_percentage

    def reset(self):
        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.reset_flow()

    def reset_flow(self):
        self.flow = 0.0
        self.cost = self.fft


class Demand:
    def __init__(self,
                 init_node: str,
                 term_node: str,
                 demand: float
                 ):
        self.fromZone = init_node
        self.toNode = term_node
        self.demand = float(demand)


def DijkstraHeap(origin, network: FlowTransportNetwork):
    """
    计算从一个起点到所有其他目的地的最短路径。
    标签和前驱存储在节点实例中。
    """
    for n in network.nodeSet:
        network.nodeSet[n].label = np.inf
        network.nodeSet[n].pred = None
    network.nodeSet[origin].label = 0.0
    network.nodeSet[origin].pred = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPred = currentNode
            existingLabel = network.nodeSet[newNode].label
            newLabel = currentLabel + network.linkSet[link].cost
            if newLabel < existingLabel:
                heapq.heappush(SE, (newLabel, newNode))
                network.nodeSet[newNode].label = newLabel
                network.nodeSet[newNode].pred = newPred


def BPRcostFunction(optimal: bool,
                    fft: float,
                    alpha: float,
                    flow: float,
                    capacity: float,
                    beta: float,
                    length: float,
                    maxSpeed: float
                    ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return fft * (1 + (alpha * math.pow((flow * 1.0 / capacity), beta)) * (beta + 1))
    return fft * (1 + alpha * math.pow((flow * 1.0 / capacity), beta))


def constantCostFunction(optimal: bool,
                         fft: float,
                         alpha: float,
                         flow: float,
                         capacity: float,
                         beta: float,
                         length: float,
                         maxSpeed: float
                         ) -> float:
    if optimal:
        return fft + flow
    return fft


def greenshieldsCostFunction(optimal: bool,
                             fft: float,
                             alpha: float,
                             flow: float,
                             capacity: float,
                             beta: float,
                             length: float,
                             maxSpeed: float
                             ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return (length * (capacity ** 2)) / (maxSpeed * (capacity - flow) ** 2)
    return length / (maxSpeed * (1 - (flow / capacity)))


def updateTravelTime(network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    此方法使用当前流量更新链接上的行驶时间
    """
    for l in network.linkSet:
        network.linkSet[l].cost = costFunction(optimal,
                                               network.linkSet[l].fft,
                                               network.linkSet[l].alpha,
                                               network.linkSet[l].flow,
                                               network.linkSet[l].capacity,
                                               network.linkSet[l].beta,
                                               network.linkSet[l].length,
                                               network.linkSet[l].speedLimit
                                               )


def findAlpha(x_bar, network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    使用无约束优化计算算法所需的最优步长
    """

    def df(alpha):
        assert 0 <= alpha <= 1
        sum_derivative = 0  # 此行为目标函数的导数。
        for l in network.linkSet:
            tmpFlow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow
            tmpCost = costFunction(optimal,
                                   network.linkSet[l].fft,
                                   network.linkSet[l].alpha,
                                   tmpFlow,
                                   network.linkSet[l].capacity,
                                   network.linkSet[l].beta,
                                   network.linkSet[l].length,
                                   network.linkSet[l].speedLimit
                                   )
            sum_derivative = sum_derivative + (x_bar[l] - network.linkSet[l].flow) * tmpCost
        return sum_derivative

    sol = scipy.optimize.root_scalar(df, x0=np.array([0.5]), bracket=(0, 1))
    assert 0 <= sol.root <= 1
    return sol.root


def tracePreds(dest, network: FlowTransportNetwork):
    """
    此方法遍历前驱节点以创建最短路径
    """
    prevNode = network.nodeSet[dest].pred
    spLinks = []
    while prevNode is not None:
        spLinks.append((prevNode, dest))
        dest = prevNode
        prevNode = network.nodeSet[dest].pred
    return spLinks


def loadAON(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    此方法为全或无加载产生辅助流量。
    """
    x_bar = {l: 0.0 for l in network.linkSet}
    SPTT = 0.0
    for r in network.originZones:
        DijkstraHeap(r, network=network)
        for s in network.zoneSet[r].destList:
            dem = network.tripSet[r, s].demand

            if dem <= 0:
                continue

            SPTT = SPTT + network.nodeSet[s].label * dem

            if computeXbar and r != s:
                for spLink in tracePreds(s, network):
                    x_bar[spLink] = x_bar[spLink] + dem

    return SPTT, x_bar


def readDemand(demand_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in demand_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        demand = row["demand"]

        network.tripSet[init_node, term_node] = Demand(init_node, term_node, demand)
        if init_node not in network.zoneSet:
            network.zoneSet[init_node] = Zone(init_node)
        if term_node not in network.zoneSet:
            network.zoneSet[term_node] = Zone(term_node)
        if term_node not in network.zoneSet[init_node].destList:
            network.zoneSet[init_node].destList.append(term_node)

    print(len(network.tripSet), "OD对")
    print(len(network.zoneSet), "OD区域")


def readNetwork(network_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in network_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        capacity = row["capacity"]
        length = row["length"]
        free_flow_time = row["free_flow_time"]
        b = row["b"]
        power = row["power"]
        speed = row["speed"]
        toll = row["toll"]
        link_type = row["link_type"]

        network.linkSet[init_node, term_node] = Link(init_node=init_node,
                                                     term_node=term_node,
                                                     capacity=capacity,
                                                     length=length,
                                                     fft=free_flow_time,
                                                     b=b,
                                                     power=power,
                                                     speed_limit=speed,
                                                     toll=toll,
                                                     linkType=link_type
                                                     )
        if init_node not in network.nodeSet:
            network.nodeSet[init_node] = Node(init_node)
        if term_node not in network.nodeSet:
            network.nodeSet[term_node] = Node(term_node)
        if term_node not in network.nodeSet[init_node].outLinks:
            network.nodeSet[init_node].outLinks.append(term_node)
        if init_node not in network.nodeSet[term_node].inLinks:
            network.nodeSet[term_node].inLinks.append(init_node)

    print(len(network.nodeSet), "节点")
    print(len(network.linkSet), "链接")


def get_TSTT(network: FlowTransportNetwork, costFunction=BPRcostFunction, use_max_capacity: bool = True):
    TSTT = round(sum([network.linkSet[a].flow * costFunction(optimal=False,
                                                             fft=network.linkSet[
                                                                 a].fft,
                                                             alpha=network.linkSet[
                                                                 a].alpha,
                                                             flow=network.linkSet[
                                                                 a].flow,
                                                             capacity=network.linkSet[
                                                                 a].max_capacity if use_max_capacity else network.linkSet[
                                                                 a].capacity,
                                                             beta=network.linkSet[
                                                                 a].beta,
                                                             length=network.linkSet[
                                                                 a].length,
                                                             maxSpeed=network.linkSet[
                                                                 a].speedLimit
                                                             ) for a in
                      network.linkSet]), 9)
    return TSTT


def assignment_loop(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    costFunction=BPRcostFunction,
                    accuracy: float = 0.001,
                    maxIter: int = 1000,
                    maxTime: int = 60,
                    verbose: bool = True):
    """
    关于算法的解释见第7章：
    https://sboyles.github.io/blubook.html
    PDF:
    https://sboyles.github.io/teaching/ce392c/book.pdf
    """
    network.reset_flow()

    iteration_number = 1
    gap = np.inf
    TSTT = np.inf
    assignmentStartTime = time.time()

    # 检查是否达到所需精度
    while gap > accuracy:

        # 通过全或无分配获取x_bar
        _, x_bar = loadAON(network=network)

        if algorithm == "MSA" or iteration_number == 1:
            alpha = (1 / iteration_number)
        elif algorithm == "FW":
            # 如果使用算法，通过解决非线性方程确定步长alpha
            alpha = findAlpha(x_bar,
                              network=network,
                              optimal=systemOptimal,
                              costFunction=costFunction)
        else:
            print("终止程序.....")
            print("解决算法 ", algorithm, " 不存在！")
            raise TypeError('算法必须是MSA或FW')

        # 应用流量改进
        for l in network.linkSet:
            network.linkSet[l].flow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow

        # 计算新的行驶时间
        updateTravelTime(network=network,
                         optimal=systemOptimal,
                         costFunction=costFunction)

        # 计算相对差距
        SPTT, _ = loadAON(network=network, computeXbar=False)
        SPTT = round(SPTT, 9)
        TSTT = round(sum([network.linkSet[a].flow * network.linkSet[a].cost for a in
                          network.linkSet]), 9)

        # print(TSTT, SPTT, "TSTT, SPTT, 最大容量", max([l.capacity for l in network.linkSet.values()]))
        gap = (TSTT / SPTT) - 1
        if gap < 0:
            print("错误，差距小于0，这不应该发生")
            print("TSTT", "SPTT", TSTT, SPTT)

            # 取消注释以调试

            # print("容量:", [l.capacity for l in network.linkSet.values()])
            # print("流量:", [l.flow for l in network.linkSet.values()])

        # 计算真实的总行驶时间（在系统最优路由的情况下与上面的TSTT不同）
        TSTT = get_TSTT(network=network, costFunction=costFunction)

        iteration_number += 1
        if iteration_number > maxIter:
            if verbose:
                print(
                    "分配未能收敛到所需的差距，已达到最大迭代次数")
                print("分配花费", round(time.time() - assignmentStartTime, 5), "秒")
                print("当前差距:", round(gap, 5))
            return TSTT
        if time.time() - assignmentStartTime > maxTime:
            if verbose:
                print("分配未能收敛到所需的差距，已达到最大时间限制")
                print("分配进行了 ", iteration_number, "次迭代")
                print("当前差距:", round(gap, 5))
            return TSTT

    if verbose:
        print("分配在 ", iteration_number, "次迭代中收敛")
        print("分配花费", round(time.time() - assignmentStartTime, 5), "秒")
        print("当前差距:", round(gap, 5))

    return TSTT


def writeResults(network: FlowTransportNetwork, output_file: str, costFunction=BPRcostFunction,
                 systemOptimal: bool = False, verbose: bool = True):
    outFile = open(output_file, "w")
    TSTT = get_TSTT(network=network, costFunction=costFunction)
    if verbose:
        print("\n总系统行驶时间:", f'{TSTT} 秒')
    tmpOut = "总行驶时间:\t" + str(TSTT)
    outFile.write(tmpOut + "\n")
    tmpOut = "使用的成本函数:\t" + BPRcostFunction.__name__
    outFile.write(tmpOut + "\n")
    tmpOut = ["用户均衡 (UE) 或系统最优 (SO):\t"] + ["SO" if systemOptimal else "UE"]
    outFile.write("".join(tmpOut) + "\n\n")
    tmpOut = "起始节点\t终止节点\t流量\t行驶时间"
    outFile.write(tmpOut + "\n")
    for i in network.linkSet:
        tmpOut = str(network.linkSet[i].init_node) + "\t" + str(
            network.linkSet[i].term_node) + "\t" + str(
            network.linkSet[i].flow) + "\t" + str(costFunction(False,
                                                               network.linkSet[i].fft,
                                                               network.linkSet[i].alpha,
                                                               network.linkSet[i].flow,
                                                               network.linkSet[i].max_capacity,
                                                               network.linkSet[i].beta,
                                                               network.linkSet[i].length,
                                                               network.linkSet[i].speedLimit
                                                               ))
        outFile.write(tmpOut + "\n")
    outFile.close()


def load_network(net_file: str,
                 demand_file: str = None,
                 force_net_reprocess: bool = False,
                 verbose: bool = True
                 ) -> FlowTransportNetwork:
    readStart = time.time()

    if demand_file is None:
        demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])

    net_name = net_file.split("/")[-1].split("_")[0]

    if verbose:
        print(f"正在加载网络 {net_name}...")

    net_df, demand_df = import_network(
        net_file,
        demand_file,
        force_reprocess=force_net_reprocess
    )

    network = FlowTransportNetwork()

    readDemand(demand_df, network=network)
    readNetwork(net_df, network=network)

    network.originZones = set([k[0] for k in network.tripSet])

    if verbose:
        print("网络", net_name, "已加载")
        print("读取网络数据花费", round(time.time() - readStart, 2), "秒\n")

    return network


def computeAssingment(net_file: str,
                      demand_file: str = None,
                      algorithm: str = "FW",  # FW 或 MSA
                      costFunction=BPRcostFunction,
                      systemOptimal: bool = False,
                      accuracy: float = 0.0001,
                      maxIter: int = 1000,
                      maxTime: int = 60,
                      results_file: str = None,
                      force_net_reprocess: bool = False,
                      verbose: bool = True
                      ) -> float:
    """
    这是计算用户均衡UE（默认）或系统最优SO交通分配的主要函数
    可以加载所有在https://github.com/bstabler/TransportationNetworks上的网络，遵循tntp格式

    :param net_file: 网络文件的名称，遵循tntp格式（见https://github.com/bstabler/TransportationNetworks）
    :param demand_file: 需求（行程）文件的名称，遵循tntp格式（见https://github.com/bstabler/TransportationNetworks），如果留空则使用默认需求文件
    :param algorithm:
           - "FW": 算法（见https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm）
           - "MSA": 连续平均法
           有关算法的更多信息见https://sboyles.github.io/teaching/ce392c/book.pdf
    :param costFunction: 用于计算边缘行驶时间的成本函数，当前可用的函数有：
           - BPRcostFunction（见https://rdrr.io/rforge/travelr/man/bpr.function.html）
           - greenshieldsCostFunction（见Greenshields, B. D., et al. "A study of traffic capacity." Highway research board proceedings. Vol. 1935. National Research Council (USA), Highway Research Board, 1935.）
           - constantCostFunction
    :param systemOptimal: 是否计算系统最优流量而不是用户均衡
    :param accuracy: 期望的分配精度差距
    :param maxIter: 算法迭代的最大次数
    :param maxTime: 分配允许的最大秒数
    :param results_file: 用于写入结果的文件名，
           默认情况下，结果文件与输入网络文件同名，后缀为"_flow.tntp"，保存在同一文件夹中
    :param force_net_reprocess: 如果应该重新处理tntp源中的网络文件，则为True
    :param verbose: 在标准输出中打印有用的信息
    :return: 总系统行驶时间
    """

    network = load_network(net_file=net_file, demand_file=demand_file, verbose=verbose, force_net_reprocess=force_net_reprocess)

    if verbose:
        print("正在计算分配...")
    TSTT = assignment_loop(network=network, algorithm=algorithm, systemOptimal=systemOptimal, costFunction=costFunction,
                           accuracy=accuracy, maxIter=maxIter, maxTime=maxTime, verbose=verbose)

    if results_file is None:
        results_file = '_'.join(net_file.split("_")[:-1] + ["flow.tntp"])

    writeResults(network=network,
                 output_file=results_file,
                 costFunction=costFunction,
                 systemOptimal=systemOptimal,
                 verbose=verbose)

    return TSTT


if __name__ == '__main__':

    # 这是一个示例用法，用于计算算法的系统最优和用户均衡

    net_file = str(PathUtils.sioux_falls_net_file)

    total_system_travel_time_optimal = computeAssingment(net_file=net_file,
                                                         algorithm="FW",
                                                         costFunction=BPRcostFunction,
                                                         systemOptimal=True,
                                                         verbose=True,
                                                         accuracy=0.00001,
                                                         maxIter=1000,
                                                         maxTime=6000000)

    total_system_travel_time_equilibrium = computeAssingment(net_file=net_file,
                                                             algorithm="FW",
                                                             costFunction=BPRcostFunction,
                                                             systemOptimal=False,
                                                             verbose=True,
                                                             accuracy=0.001,
                                                             maxIter=1000,
                                                             maxTime=6000000)

    print("UE - SO = ", total_system_travel_time_equilibrium - total_system_travel_time_optimal)