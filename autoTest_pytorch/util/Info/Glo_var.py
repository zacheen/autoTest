from .Game_Env import init_game_env
from .Game_State import init_game_state

class Glo_var:
    """Session-wide data hub.

    Composition over inheritance:
      • config  — static (paths, credentials, labels)
      • state   — dynamic (counters, flags, buffers)
      • session — I/O resources (logs, driver, backend)

    Direct attribute access is fine — no getter/setter logic. Call sites read
    e.g. `glo_var.state.round_count`, `glo_var.session.backend`.
    """

    def __init__(self, game_name, game_env, player_num, round_count,
                 suit_order=None, list_len=3):
        self.env   = init_game_env(game_name, game_env)
        self.state = init_game_state(round_count, player_num, list_len)

    # Thin convenience delegates — common enough to keep at the top level
    def reset(self, round_count):
        self.state.reset(round_count)