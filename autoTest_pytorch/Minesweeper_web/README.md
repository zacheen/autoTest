# Minesweeper Web

This folder adds a browser UI for the existing `MinesweeperLogic` backend.

## Run

From the project root, using your miniconda environment:

```bash
conda activate <your-env>
pip install -r requirements.txt
python autoTest_pytorch/Minesweeper_web/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

If you prefer calling miniconda Python directly:

```bash
C:\Users\User\miniconda3\python.exe autoTest_pytorch\Minesweeper_web\server.py
```

## API

- `GET /api/difficulties`
- `POST /api/games`
- `GET /api/games/<game_id>`
- `POST /api/games/<game_id>/click`
- `POST /api/games/<game_id>/flag`

## Notes

- The backend keeps each game session in memory.
- The first left click starts the timer.
- The layout is modeled after the Tkinter desktop UI: toolbar, info row, centered board.
