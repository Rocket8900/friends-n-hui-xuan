from flask import Blueprint, request, jsonify
import random

fog_bp = Blueprint("fog_bp", __name__)

maps = {}
crow_positions = {}

@fog_bp.post("/fog-of-wall")
def fog_of_wall():
    data = request.get_json(force=True)
    challenger_id = data.get("challenger_id")
    game_id = data.get("game_id")

    # ----- Initial test case request -----
    if "test_case" in data:
        test_case = data["test_case"]
        length = test_case["length_of_grid"]
        maps[game_id] = set()  # discovered walls initially empty

        # Initialize crow positions
        crow_positions[game_id] = {c["id"]: (c["x"], c["y"]) for c in test_case["crows"]}

        # Choose first crow (pick the first one for deterministic behavior)
        first_crow = test_case["crows"][0]

        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": first_crow["id"],
            "action_type": "scan"
        })

    # ----- Subsequent move or scan -----
    previous_action = data.get("previous_action")
    if not previous_action:
        return jsonify({"error": "No previous_action provided"}), 400

    crow_id = previous_action.get("crow_id")

    # Handle move
    if previous_action.get("your_action") == "move":
        move_result = previous_action.get("move_result")
        crow_positions[game_id][crow_id] = tuple(move_result)

        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": crow_id,
            "action_type": "move",
            "direction": previous_action.get("direction")
        })

    # Handle scan
    elif previous_action.get("your_action") == "scan":
        scan_result = previous_action.get("scan_result")
        cx, cy = 2, 2  # scan grid centered on crow

        # Update discovered walls
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                val = scan_result[dy + 2][dx + 2]
                if val == "W":
                    x = cx + dx
                    y = cy + dy
                    maps[game_id].add(f"{x}-{y}")

        # Submit all discovered walls
        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": list(maps[game_id])
        })

    return jsonify({"error": "Unknown previous_action"}), 400
