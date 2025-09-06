from flask import Blueprint, request, jsonify

bureau_bp = Blueprint("bureau_bp", __name__)

def find_extra_channels(edges):
    parent = {}

    def find(u):
        while parent.get(u, u) != u:
            u = parent.get(u, u)
        return u

    extra = []
    for edge in edges:
        u = edge["spy1"]
        v = edge["spy2"]
        pu = find(u)
        pv = find(v)
        if pu == pv:
            # adding this edge creates a cycle
            extra.append(edge)
        else:
            parent[pu] = pv
    return extra

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