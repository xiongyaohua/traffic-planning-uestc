import pandas as pd
import numpy as np
import openmatrix as omx

from utils import PathUtils


def import_network(network_file: str, demand_file: str, force_reprocess: bool = False):
    """
    此方法从相应的tntp文件中导入网络和需求（见https://github.com/bstabler/TransportationNetworks）
    导入后，它会将它们存储在输入文件相同的目录中的更快格式中，
    如果再次调用此方法，它将自动访问已转换的文件

    :param network_file: 网络（net）文件名
    :param demand_file: 需求（trips）文件名
    :param force_reprocess: 如果应从tntp文件重新处理网络，则为True
    :return: None
    """

    network_file_csv = network_file.split(".")[0].split("/")[-1] + ".csv"
    demand_file_csv = demand_file.split(".")[0].split("/")[-1] + ".csv"

    network_file_csv = PathUtils.processed_networks_folder / network_file_csv
    demand_file_csv = PathUtils.processed_networks_folder / demand_file_csv

    if network_file_csv.is_file() and not force_reprocess:
        net_df = pd.read_csv(str(network_file_csv),
                             sep='\t')
    else:
        net_df = _net_file2df(network_file)
        net_df.to_csv(path_or_buf=str(network_file_csv),
                      sep='\t',
                      index=False)

    if demand_file_csv.is_file() and not force_reprocess:
        demand_df = pd.read_csv(str(demand_file_csv),
                                sep='\t')
    else:
        tripSet = _demand_file2trips(demand_file)
        demand_df = pd.DataFrame(columns=["init_node", "term_node", "demand"])

        k = 0
        for orig in tripSet:
            for dest in tripSet[orig]:
                demand_df.loc[k] = [orig, dest, tripSet[orig][dest]]
                k += 1

        demand_df = demand_df.astype({"init_node": int, "term_node": int})

        demand_df.to_csv(path_or_buf=str(demand_file_csv),
                         sep='\t',
                         index=False)

    return net_df, demand_df


def _net_file2df(network_file: str):
    net_df = pd.read_csv(network_file, skiprows=8, sep='\t')

    trimmed_columns = [s.strip().lower() for s in net_df.columns]
    net_df.columns = trimmed_columns

    net_df.drop(['~', ';'], axis=1, inplace=True)
    return net_df

def _demand_file2trips(demand_file: str):
    """
    从需求文件中解析行程数据
    """
    f = open(demand_file, 'r')
    all_rows = f.read()
    f.close()
    blocks = all_rows.split('Origin')[1:]
    tripSet = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig = int(orig[0])
        d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        tripSet[orig] = destinations

    return tripSet

def _demand_file2matrix(demand_file: str, omx_write_file_path: str = None):
    """
    将需求文件转换为矩阵并写入OMX文件
    """
    f = open(demand_file, 'r')
    all_rows = f.read()
    f.close()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig = int(orig[0])
        d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())

    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            # 将值映射到索引i-1，因为Numpy是基于0的
            mat[i, j] = matrix.get(i + 1, {}).get(j + 1, 0)

    if omx_write_file_path:
        index = np.arange(zones) + 1
        myfile = omx.open_file(omx_write_file_path, 'w')
        myfile['matrix'] = mat
        myfile.create_mapping('taz', index)
        myfile.close()

    return mat