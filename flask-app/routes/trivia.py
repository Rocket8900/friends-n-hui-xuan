from routes import app
from flask import jsonify

@app.route("/trivia", methods=["GET"])
def trivia():
    answers = [
        4, 
        1,
        2,
        2,
    ]
    return jsonify(answers)
