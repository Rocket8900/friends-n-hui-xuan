from flask import Blueprint, request, jsonify

ticketing_bp = Blueprint("ticketing_bp", __name__)

VIP_POINTS = 100
PRIORITY_CC_POINTS = 50

D2_BUCKET_1 = 2 * 2 
D2_BUCKET_2 = 4 * 4  


def latency_points_sqdist(d2: int) -> int:
    if d2 <= D2_BUCKET_1:
        return 30
    if d2 <= D2_BUCKET_2:
        return 20
    return 0


def score_customer_for_concert(customer, concert, cc_priority_map):
    score = VIP_POINTS if customer.get("vip_status", False) else 0

    if cc_priority_map.get(customer.get("credit_card")) == concert.get("name"):
        score += PRIORITY_CC_POINTS

    cx, cy = customer["location"]
    bx, by = concert["booking_center_location"]
    d2 = (cx - bx) ** 2 + (cy - by) ** 2
    score += latency_points_sqdist(d2)

    return score


@ticketing_bp.post("/ticketing-agent")
def ticketing_agent():
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    customers = payload.get("customers", [])
    concerts = payload.get("concerts", [])
    cc_priority_map = payload.get("priority", {}) or {}

    if not isinstance(customers, list) or not isinstance(concerts, list) or not isinstance(cc_priority_map, dict):
        return jsonify({"error": "Invalid payload schema"}), 400
    if not customers or not concerts:
        return jsonify({"error": "Payload must include non-empty 'customers' and 'concerts' lists"}), 400

    for c in concerts:
        if "name" not in c or "booking_center_location" not in c:
            return jsonify({"error": "Each concert needs 'name' and 'booking_center_location'"}), 400

    result = {}

    for cust in customers:
        if ("name" not in cust or
            "vip_status" not in cust or
            "location" not in cust or
            "credit_card" not in cust):
            return jsonify({"error": f"Customer entry missing required fields: {cust}"}), 400

        best_concert = None
        best_score = float("-inf")

        for con in concerts:
            score = score_customer_for_concert(cust, con, cc_priority_map)
            if score > best_score:
                best_score = score
                best_concert = con["name"]

        result[cust["name"]] = best_concert

    return jsonify(result), 200
