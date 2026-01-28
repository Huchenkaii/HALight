import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

def _split_sign(road_id: str) -> Tuple[str, bool]:
    if road_id.startswith("-"):
        return road_id[1:], True
    return road_id, False

def _validate_as_roadnet(obj: Union[dict, list]) -> dict:
    if isinstance(obj, dict) and isinstance(obj.get("intersections", None), list):
        return obj
    if isinstance(obj, list):
        raise TypeError(
            "The input file appears to be a flow file (top-level is a list), rather than a roadnet file. "
            "Please provide a road network file (top-level is a dict containing the 'intersections' key)."
        )
    raise TypeError(
        "Unrecognized JSON structure: expected a roadnet (dict containing an 'intersections' list)."
    )

def build_neighbors_from_dict(roadnet: dict, include_virtual: bool = False) -> Dict[str, List[str]]:
    """
    Build a first-order adjacency list from a CityFlow road network (roadnet dict).
    """
    intersections = roadnet.get("intersections", [])
    nodes = [it for it in intersections if include_virtual or not it.get("virtual", False)]

    node_roads: Dict[str, Set[str]] = {it["id"]: set(it.get("roads", [])) for it in nodes}

    base_buckets: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"plus": set(), "minus": set()})
    for nid, roads in node_roads.items():
        for r in roads:
            base, is_minus = _split_sign(r)
            (base_buckets[base]["minus"] if is_minus else base_buckets[base]["plus"]).add(nid)

    neighbors_set: Dict[str, Set[str]] = {nid: set() for nid in node_roads.keys()}
    for nid, roads in node_roads.items():
        for r in roads:
            base, is_minus = _split_sign(r)
            targets = base_buckets[base]["plus" if is_minus else "minus"]
            for t in targets:
                if t != nid:
                    neighbors_set[nid].add(t)

    return {nid: sorted(list(nset)) for nid, nset in neighbors_set.items()}

def build_neighbors_from_file(json_path: str, include_virtual: bool = False) -> Dict[str, List[str]]:
    """
    Build a first-order adjacency list from a CityFlow road network JSON file.
    Please provide a road network file (not a flow file).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    roadnet = _validate_as_roadnet(obj)
    return build_neighbors_from_dict(roadnet, include_virtual=include_virtual)