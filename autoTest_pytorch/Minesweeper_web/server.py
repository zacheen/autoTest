import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory


CURRENT_DIR = Path(__file__).resolve().parent
MINESWEEPER_DIR = CURRENT_DIR.parent / "Minesweeper"

if str(MINESWEEPER_DIR) not in sys.path:
    sys.path.insert(0, str(MINESWEEPER_DIR))

from MinesweeperLogic import MinesweeperLogic  # noqa: E402


app = Flask(__name__, static_folder=str(CURRENT_DIR / "static"), static_url_path="/static")
session_lock = Lock()

DIFFICULTIES = {
    "Training 6x6": {"rows": 6, "cols": 6, "mines": 6},
    "Beginner": {"rows": 9, "cols": 9, "mines": 10},
    "Intermediate": {"rows": 16, "cols": 16, "mines": 40},
    "Expert": {"rows": 16, "cols": 30, "mines": 99},
}


@dataclass
class GameSession:
    game_id: str
    difficulty: str
    logic: MinesweeperLogic
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    hit_mine: Optional[Tuple[int, int]] = None


games: Dict[str, GameSession] = {}


def _build_game(difficulty: str) -> GameSession:
    params = DIFFICULTIES[difficulty]
    return GameSession(
        game_id=str(uuid.uuid4()),
        difficulty=difficulty,
        logic=MinesweeperLogic(params["rows"], params["cols"], params["mines"]),
        created_at=time.time(),
    )


def _get_session_or_404(game_id: str) -> GameSession:
    game = games.get(game_id)
    if game is None:
        raise KeyError(game_id)
    return game


def _status_for(game: GameSession) -> str:
    if game.logic.game_over:
        return "lost"
    if game.logic.is_win:
        return "won"
    if game.started_at is None:
        return "ready"
    return "playing"


def _elapsed_seconds(game: GameSession) -> int:
    if game.started_at is None:
        return 0
    if game.finished_at is not None:
        return int(game.finished_at - game.started_at)
    return int(time.time() - game.started_at)


def _serialize_cell(game: GameSession, row: int, col: int) -> dict:
    coord = (row, col)
    logic = game.logic
    is_finished = logic.game_over or logic.is_win

    if not is_finished:
        if coord in logic.flags:
            return {"state": "flagged", "value": None}
        if coord in logic.revealed:
            return {"state": "revealed", "value": logic._count_adjacent_mines(row, col)}
        return {"state": "hidden", "value": None}

    if logic.is_win and coord in logic.mines:
        return {"state": "flagged_mine", "value": None}

    if logic.game_over:
        if coord == game.hit_mine:
            return {"state": "hit_mine", "value": None}
        if coord in logic.mines and coord in logic.flags:
            return {"state": "flagged_mine", "value": None}
        if coord in logic.mines:
            return {"state": "mine", "value": None}
        if coord in logic.flags and coord not in logic.mines:
            return {"state": "wrong_flag", "value": None}

    if coord in logic.revealed:
        return {"state": "revealed", "value": logic._count_adjacent_mines(row, col)}
    if coord in logic.flags:
        return {"state": "flagged", "value": None}
    return {"state": "hidden", "value": None}


def _serialize_game(game: GameSession) -> dict:
    board = []
    for row in range(game.logic.rows):
        board.append([_serialize_cell(game, row, col) for col in range(game.logic.cols)])

    return {
        "game_id": game.game_id,
        "difficulty": game.difficulty,
        "status": _status_for(game),
        "rows": game.logic.rows,
        "cols": game.logic.cols,
        "mines_count": game.logic.mines_count,
        "remaining_mines": game.logic.remaining_mines,
        "elapsed_seconds": _elapsed_seconds(game),
        "board": board,
    }


def _json_error(message: str, status_code: int):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/difficulties")
def get_difficulties():
    return jsonify({"difficulties": DIFFICULTIES})


@app.post("/api/games")
def create_game():
    payload = request.get_json(silent=True) or {}
    difficulty = payload.get("difficulty", "Beginner")

    if difficulty not in DIFFICULTIES:
        return _json_error("Unknown difficulty.", 400)

    with session_lock:
        game = _build_game(difficulty)
        games[game.game_id] = game

    response = jsonify(_serialize_game(game))
    response.status_code = 201
    return response


@app.get("/api/games/<game_id>")
def get_game(game_id: str):
    try:
        game = _get_session_or_404(game_id)
    except KeyError:
        return _json_error("Game session not found.", 404)

    return jsonify(_serialize_game(game))


@app.post("/api/games/<game_id>/click")
def click_cell(game_id: str):
    payload = request.get_json(silent=True) or {}
    row = payload.get("row")
    col = payload.get("col")

    if not isinstance(row, int) or not isinstance(col, int):
        return _json_error("row and col must be integers.", 400)

    try:
        game = _get_session_or_404(game_id)
    except KeyError:
        return _json_error("Game session not found.", 404)

    with session_lock:
        first_click = game.logic.first_click
        result = game.logic.click(row, col)
        if first_click and result.changed and game.started_at is None:
            game.started_at = time.time()
        if (result.game_over or result.win) and game.finished_at is None:
            game.finished_at = time.time()
        if result.game_over:
            game.hit_mine = result.hit_mine

    return jsonify(_serialize_game(game))


@app.post("/api/games/<game_id>/flag")
def flag_cell(game_id: str):
    payload = request.get_json(silent=True) or {}
    row = payload.get("row")
    col = payload.get("col")

    if not isinstance(row, int) or not isinstance(col, int):
        return _json_error("row and col must be integers.", 400)

    try:
        game = _get_session_or_404(game_id)
    except KeyError:
        return _json_error("Game session not found.", 404)

    with session_lock:
        game.logic.flag(row, col)
        if game.logic.is_win and game.finished_at is None:
            game.finished_at = time.time()

    return jsonify(_serialize_game(game))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
