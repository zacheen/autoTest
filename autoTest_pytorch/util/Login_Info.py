# ── credential loaders (branch on game_envi) ───────────────────

    def _load_credentials(self):
        envi = self.game_envi
        print("Game_envi :", envi)
        if isinstance(envi, str) and envi.startswith("url"):
            self.is_url = True
            print("Game_envi is url")
            self._load_url_credentials()
        elif envi == "Minesweeper_local_py":
            self._load_file_credentials(self.user_change_path / "Minesweeper_input.txt")
        else:
            # web / other envs — no credentials file to read
            pass

    def _load_url_credentials(self):
        ip = ""
        try:
            response = requests.get(f"http://{ip}/crawler/getCompanys")
            self.DaiLi_data = response.json()
            print("response after json : " + str(self.DaiLi_data))
        except Exception:
            print("json fail so using local file")
            with open(self.user_change_path / 'url.json', encoding='UTF-8') as f:
                self.DaiLi_data = json.load(f)

    def _load_file_credentials(self, path):
        def _field(f):
            return str(f.readline().split(" -:")[1]).strip()
        with open(path, "r", encoding='UTF-8') as f:
            self.game_account    = _field(f); print("Account: "         + self.game_account)
            self.game_password   = _field(f); print("Password: "        + self.game_password)
            self.game_agent_ID   = _field(f); print("Agent ID: "        + self.game_agent_ID)
            self.game_money      = _field(f); print("Credits: "         + self.game_money)
            self.game_envir      = _field(f); print("Environment: "     + self.game_envir)
            self.server_account  = _field(f); print("Server account: "  + self.server_account)
            self.server_password = _field(f); print("Server password: " + self.server_password)

    def _load_class_to_str_list(self):
        try:
            return Data.name_list[self.game_name]
        except KeyError:
            print(f"Warning: {self.game_name} not found in Data.py")
            return {}