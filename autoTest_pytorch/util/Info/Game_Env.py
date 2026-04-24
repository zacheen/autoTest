from pathlib import Path

class _Game_Env:
    """Static per-session config — paths, credentials, label map.

    Read-only after init. Owns:
      • game_name, list_len, game_env, is_url
      • user_change_path, game_pic_path, cut_pic_path
      • credentials (game_account / game_password / ...)
      • class_to_str_list (label → string map from Data.py)
    """
    def __init__(self):
        self.game_name = None

    def final_var(self, game_name, game_env):
        if self.game_name != None :
            raise Exception("already initialized once")
        self.game_name = game_name
        self.game_env  = game_env

        self.user_change_path = self._find_user_change_dir()
        self._setup_game_pic_paths()
        # self._load_credentials()

        # self.class_to_str_list = self._load_class_to_str_list()

    # ── setup helpers ──────────────────────────────────────────────

    @staticmethod
    def _find_user_change_dir():
        base_dir = Path("./")
        return list(base_dir.rglob("user_change"))[0]

    def _setup_game_pic_paths(self):
        (self.user_change_path / "game_pic").mkdir(exist_ok=True)
        self.game_pic_path = self.user_change_path / "game_pic" / f"{self.game_name}_pic"
        self.game_pic_path.mkdir(exist_ok=True)

        training_data_path = self.game_pic_path / "training_data"
        training_data_path.mkdir(exist_ok=True)
        self.cut_pic_path = str(training_data_path) + "\\"

        # export game name to the identification module
        import util.identify_for_import as identify_for_import
        identify_for_import.game_name = self.game_name


# Singleton
instance = _Game_Env()
def init_game_env(game_name, game_env) :
    global instance
    instance.final_var(game_name, game_env)
    return instance

def get_game_env() :
    global instance
    return instance