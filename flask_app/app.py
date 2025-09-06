import logging
import socket
from routes import app
from routes.ticketing_agent import ticketing_bp
from routes.princess_diaries import princess_bp
from routes.blankety_blanks import blankety_bp
from routes.universal_bureau import bureau_bp
from routes.fog_of_wall import fog_bp
from routes.trading_formula import trading_formula_bp
from routes.duolingo_sort import duolingo_sort_bp

logger = logging.getLogger(__name__)


@app.route("/", methods=["GET"])
def default_route():
    return "Python Template"


app.register_blueprint(ticketing_bp)
app.register_blueprint(duolingo_sort_bp)

app.register_blueprint(blankety_bp)
app.register_blueprint(bureau_bp)
app.register_blueprint(princess_bp)
app.register_blueprint(fog_bp)
app.register_blueprint(trading_formula_bp)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logging.info("Starting application ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 8080))
    port = sock.getsockname()[1]
    sock.close()
    app.run(port=port)
