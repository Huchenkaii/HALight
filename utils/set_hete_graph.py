import torch
from torch import tensor
from torch_geometric.data import HeteroData
import json

def build_hetero_graph_from_cityflow(net_path):
    with open(net_path, "r") as f:
        roadnet = json.load(f)

    hete_data = HeteroData()

    intersections = roadnet["intersections"]
    roads = roadnet["roads"]

    raw_T_nodes = []
    raw_Cross_nodes = []

    for inter in intersections:
        if inter.get("virtual", True):
            continue  # 跳过虚拟节点

        iid = inter["id"]
        num_phases = len(inter.get("trafficLight", {}).get("lightphases", []))

        if num_phases == 3:
            raw_T_nodes.append(iid)
        elif num_phases == 4:
            raw_Cross_nodes.append(iid)
        else:
            print(f"[warning] Node {iid} has unexpected number of lightphases: {num_phases}")

    T_nodes = sorted(raw_T_nodes, key=int)
    Cross_nodes = sorted(raw_Cross_nodes, key=int)

    ts_id2graph_id_T = {int(iid): int(idx) for idx, iid in enumerate(T_nodes)}
    graph_id2ts_id_T = {int(idx): int(iid) for idx, iid in enumerate(T_nodes)}

    ts_id2graph_id_Cross = {int(iid): int(idx) for idx, iid in enumerate(Cross_nodes)}
    graph_id2ts_id_Cross = {int(idx): int(iid) for idx, iid in enumerate(Cross_nodes)}

    # 创建节点特征张量
    hete_data['T'].x = torch.zeros(len(T_nodes), 13)
    hete_data['Cross'].x = torch.zeros(len(Cross_nodes), 25)

    # 构建边集合
    T2T_edges = []
    T2Cross_edges = []
    Cross2T_edges = []
    Cross2Cross_edges = []

    for road in roads:
        start_raw = road["startIntersection"]
        end_raw = road["endIntersection"]

        # 确保 start 和 end 是非虚拟节点（即出现在 T_nodes 或 Cross_nodes 中）
        is_start_valid = start_raw in T_nodes or start_raw in Cross_nodes
        is_end_valid = end_raw in T_nodes or end_raw in Cross_nodes

        if not is_start_valid or not is_end_valid:
            continue

        # 转成 int
        start = int(start_raw)
        end = int(end_raw)

        # 边分类
        if start in ts_id2graph_id_T and end in ts_id2graph_id_T:
            T2T_edges.append([ts_id2graph_id_T[start], ts_id2graph_id_T[end]])
        elif start in ts_id2graph_id_T and end in ts_id2graph_id_Cross:
            T2Cross_edges.append([ts_id2graph_id_T[start], ts_id2graph_id_Cross[end]])
        elif start in ts_id2graph_id_Cross and end in ts_id2graph_id_T:
            Cross2T_edges.append([ts_id2graph_id_Cross[start], ts_id2graph_id_T[end]])
        elif start in ts_id2graph_id_Cross and end in ts_id2graph_id_Cross:
            Cross2Cross_edges.append([ts_id2graph_id_Cross[start], ts_id2graph_id_Cross[end]])

    # 使用集合去重，确保无复边
    T2T_edges_set = set(map(tuple, T2T_edges))
    T2Cross_edges_set = set(map(tuple, T2Cross_edges))
    Cross2T_edges_set = set(map(tuple, Cross2T_edges))
    Cross2Cross_edges_set = set(map(tuple, Cross2Cross_edges))

    # 构造 PyG 的 edge_index（确保无复边，方向性保留）
    if T2T_edges_set:
        T2T_tensor = tensor(list(T2T_edges_set), dtype=torch.long).t().contiguous()
        hete_data['T', 'link', 'T'].edge_index = T2T_tensor

    if T2Cross_edges_set:
        T2Cross_tensor = tensor(list(T2Cross_edges_set), dtype=torch.long).t().contiguous()
        hete_data['T', 'link', 'Cross'].edge_index = T2Cross_tensor

    if Cross2T_edges_set:
        Cross2T_tensor = tensor(list(Cross2T_edges_set), dtype=torch.long).t().contiguous()
        hete_data['Cross', 'link', 'T'].edge_index = Cross2T_tensor

    if Cross2Cross_edges_set:
        Cross2Cross_tensor = tensor(list(Cross2Cross_edges_set), dtype=torch.long).t().contiguous()
        hete_data['Cross', 'link', 'Cross'].edge_index = Cross2Cross_tensor

    T_nodes = [int(x) for x in T_nodes]
    Cross_nodes = [int(x) for x in Cross_nodes]

    return hete_data, T_nodes, Cross_nodes, ts_id2graph_id_T, ts_id2graph_id_Cross, graph_id2ts_id_T, graph_id2ts_id_Cross

# 示例主程序（可选）
if __name__ == '__main__':
    path = '../Nets/5x5/5x5.json'  # 替换为你的路径
    hete_data, T_nodes, Cross_nodes, ts_id2graph_id_T, ts_id2graph_id_Cross, graph_id2ts_id_T, graph_id2ts_id_Cross = build_hetero_graph_from_cityflow(path)
    print(T_nodes)
    print(hete_data)
