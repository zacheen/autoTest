class MinesweeperWebClient:
    """Small client for interacting with the Minesweeper web API via Selenium."""

    def __init__(self, default_difficulty="Training 6x6"):
        self.default_difficulty = default_difficulty

    def _get_driver(self):
        import Tool_Main

        driver = getattr(Tool_Main.glo_var, "game_driver", None)
        if driver is None:
            print("[MinesweeperWebClient] Browser driver not ready")
            return None
        return driver

    def click_cell(self, row, col):
        driver = self._get_driver()
        if driver is None:
            return False

        try:
            result = driver.execute_async_script(
                """
                const row = arguments[0];
                const col = arguments[1];
                const difficulty = arguments[2];
                const done = arguments[arguments.length - 1];
                let gameId = window.localStorage.getItem('minesweeper-web-game-id');

                const ensureGame = gameId
                    ? Promise.resolve(gameId)
                    : fetch('/api/games', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({difficulty}),
                    })
                        .then(resp => resp.json())
                        .then(data => {
                            if (!data.game_id) {
                                throw new Error('Failed to create game');
                            }
                            window.localStorage.setItem('minesweeper-web-game-id', data.game_id);
                            return data.game_id;
                        });

                ensureGame
                    .then(id => fetch(`/api/games/${id}/click`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({row, col}),
                    }))
                    .then(resp => resp.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        done({ok: true, data});
                    })
                    .catch(err => done({ok: false, error: String(err)}));
                """,
                int(row),
                int(col),
                self.default_difficulty,
            )

            if result and result.get("ok"):
                driver.refresh()
                return True

            print(f"[MinesweeperWebClient] Click API failed: {result}")
            return False
        except Exception as e:
            print(f"[MinesweeperWebClient] Click API exception: {e}")
            return False

    def start_new_game(self, difficulty=None):
        driver = self._get_driver()
        if driver is None:
            return False

        difficulty = difficulty or self.default_difficulty

        try:
            result = driver.execute_async_script(
                """
                const difficulty = arguments[0];
                const done = arguments[arguments.length - 1];

                fetch('/api/games', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({difficulty}),
                })
                    .then(resp => resp.json())
                    .then(data => {
                        if (!data.game_id) {
                            throw new Error('Failed to create new game');
                        }
                        window.localStorage.setItem('minesweeper-web-game-id', data.game_id);
                        done({ok: true, data});
                    })
                    .catch(err => done({ok: false, error: String(err)}));
                """,
                difficulty,
            )

            if result and result.get("ok"):
                driver.refresh()
                return True

            print(f"[MinesweeperWebClient] New game API failed: {result}")
            return False
        except Exception as e:
            print(f"[MinesweeperWebClient] New game API exception: {e}")
            return False
