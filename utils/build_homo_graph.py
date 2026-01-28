import torch
from torch_geometric.data import Data
import json


def build_homo_graph_from_cityflow(net_path, feature_dim=25):
    """
    Build a homogeneous graph from a CityFlow road network JSON
    (all nodes share the same embedding space).
    """
    with open(net_path, "r") as f:
        roadnet = json.load(f)

    intersections = roadnet["intersections"]
    roads = roadnet["roads"]

    # --- Valid nodes: non-virtual nodes with traffic lights ---

    valid_nodes = sorted([
        inter["id"]
        for inter in intersections
        if not inter.get("virtual", True) and "trafficLight" in inter
    ], key=int)

    ts_id2graph_id = {int(iid): idx for idx, iid in enumerate(valid_nodes)}
    graph_id2ts_id = {idx: int(iid) for iid, idx in ts_id2graph_id.items()}

    x = torch.zeros(len(valid_nodes), feature_dim)

    edges = []
    for road in roads:
        start = road["startIntersection"]
        end = road["endIntersection"]

        if start not in valid_nodes or end not in valid_nodes:
            continue

        start_id = ts_id2graph_id[int(start)]
        end_id = ts_id2graph_id[int(end)]

        edges.append([start_id, end_id])
        edges.append([end_id, start_id])

    if len(edges) == 0:
        raise ValueError("No valid edges found in the road network.")

    edge_index = torch.tensor(list(set(map(tuple, edges))), dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)

    return data, valid_nodes, ts_id2graph_id, graph_id2ts_id

