"""One-hop neighbor construction for CityFlow roadnet files.

This implementation does NOT depend on road-id naming conventions such as
E/-E.  It builds neighbors directly from each road/edge's endpoints:
startIntersection/endIntersection.  For extra compatibility, it also accepts
from/to or fromNode/toNode when those fields are used by another converter.
"""

import json
from typing import Dict, List, Set, Union, Any, Optional


def _validate_as_roadnet(obj: Union[dict, list]) -> dict:
    """
    Validate and return a roadnet dict.

    Expected CityFlow-style roadnet:
        {"intersections": [...], "roads": [...]}

    A flow file is usually a top-level list, so it is rejected explicitly.
    """
    if isinstance(obj, dict) and isinstance(obj.get("intersections", None), list):
        return obj
    if isinstance(obj, list):
        raise TypeError(
            "输入文件看起来是 flow（顶层为 list），而不是 roadnet。"
            " 请改为传入路网文件（顶层为 dict，包含 'intersections' 键）。"
        )
    raise TypeError(
        "无法识别的 JSON 结构：期望 roadnet（dict，含 'intersections' 列表）。"
    )


def _sort_key(node_id: Any):
    """Sort numeric ids numerically and other ids lexicographically."""
    s = str(node_id)
    return (0, int(s)) if s.isdigit() else (1, s)


def _get_edge_endpoint(edge: dict, start: bool = True) -> Optional[str]:
    """
    Read an edge endpoint from common roadnet field names.

    CityFlow roadnet normally uses:
        startIntersection / endIntersection

    Some converters may use:
        from / to, fromNode / toNode, from_node / to_node
    """
    if start:
        candidates = ("startIntersection", "from", "fromNode", "from_node")
    else:
        candidates = ("endIntersection", "to", "toNode", "to_node")

    for key in candidates:
        if key in edge and edge[key] is not None:
            return str(edge[key])
    return None


def _get_edges(roadnet: dict) -> List[dict]:
    """
    Return road/edge list.

    CityFlow uses 'roads'.  The fallback 'edges' is kept for compatibility with
    other roadnet converters.
    """
    roads = roadnet.get("roads", None)
    if isinstance(roads, list):
        return roads

    edges = roadnet.get("edges", None)
    if isinstance(edges, list):
        return edges

    return []


def build_neighbors_from_dict(
    roadnet: dict,
    include_virtual: bool = False,
    directed: bool = False,
) -> Dict[str, List[str]]:
    """
    Build a one-hop neighbor table from a roadnet dict.

    Parameters
    ----------
    roadnet:
        CityFlow-style roadnet dictionary.
    include_virtual:
        If False, virtual boundary intersections are excluded.
    directed:
        If False, an existing road u -> v makes u and v mutual neighbors.
        If True, only u -> v is added.

    Returns
    -------
    Dict[str, List[str]]
        Example:
        {
            "0": ["1", "3"],
            "1": ["0", "2", "4"],
        }
    """
    intersections = roadnet.get("intersections", [])

    valid_nodes: Set[str] = {
        str(it["id"])
        for it in intersections
        if "id" in it and (include_virtual or not it.get("virtual", False))
    }

    neighbors_set: Dict[str, Set[str]] = {nid: set() for nid in valid_nodes}

    for edge in _get_edges(roadnet):
        start = _get_edge_endpoint(edge, start=True)
        end = _get_edge_endpoint(edge, start=False)

        if start is None or end is None:
            continue
        if start == end:
            continue
        if start not in valid_nodes or end not in valid_nodes:
            continue

        neighbors_set[start].add(end)
        if not directed:
            neighbors_set[end].add(start)

    return {
        nid: sorted(list(nset), key=_sort_key)
        for nid, nset in sorted(neighbors_set.items(), key=lambda kv: _sort_key(kv[0]))
    }


def build_neighbors_from_file(
    json_path: str,
    include_virtual: bool = False,
    directed: bool = False,
) -> Dict[str, List[str]]:
    """
    Build a one-hop neighbor table from a roadnet JSON file.

    注意：这里应传入 roadnet 文件，而不是 flow 文件。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    roadnet = _validate_as_roadnet(obj)
    return build_neighbors_from_dict(
        roadnet,
        include_virtual=include_virtual,
        directed=directed,
    )


if __name__ == "__main__":
    # Example:
    # python get_neighbour.py ../Nets/4x4/4x4.json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python get_neighbour.py <roadnet_json>")
        sys.exit(0)

    neighbors = build_neighbors_from_file(sys.argv[1], include_virtual=False)
    for node_id, neighs in neighbors.items():
        print(f"{node_id}: {neighs}")
