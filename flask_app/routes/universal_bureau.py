from flask import Blueprint, request, jsonify
import networkx as nx

bureau_bp = Blueprint("bureau_bp", __name__)

def find_extra_channels(edges):
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["spy1"], e["spy2"])

    # Find all simple cycles (lists of nodes)
    cycles = nx.cycle_basis(G)

    # Collect all edges from all cycles
    edge_set = set()
    for cycle in cycles:
        L = len(cycle)
        for i in range(L):
            u, v = cycle[i], cycle[(i + 1) % L]
            edge_set.add(tuple(sorted([u, v])))

    # Return original edges that are part of any cycle
    extra_edges = [e for e in edges if tuple(sorted([e["spy1"], e["spy2"]])) in edge_set]
    return extra_edges

@bureau_bp.post("/investigate")
def investigate():
    try:
        payload = request.get_json(force=True)
        if not payload or "networks" not in payload:
            return jsonify({"error": "Expected key 'networks'"}), 400

        result = {"networks": []}
        for net in payload["networks"]:
            net_id = net.get("networkId")
            edges = net.get("network", [])
            extra = find_extra_channels(edges)
            result["networks"].append({
                "networkId": net_id,
                "extraChannels": extra
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500