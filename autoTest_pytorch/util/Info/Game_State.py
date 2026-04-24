import datetime

class _GameState:
    """Dynamic runtime state — mutates during play.

    Call reset(round_count) to re-init for a new playthrough (e.g. after an
    error restart). Owns:
      • round counters (round_count, round_count_for_pipe)
      • ring-buffered per-round data (client_data, begin_time, end_time)
      • flags (fail_playing, server_using)
      • transient values (mid_pos, record_time, file_create_time)
    """

    def __init__(self):
        self.player_num = None

    def final_var(self, round_count, player_num, list_len) :
        if self.player_num != None :
            raise Exception("already initialized once")
        self.player_num = player_num
        self.list_len   = list_len
        self.reset(round_count)

    def reset(self, round_count):
        # ring buffer: one slot per outstanding round (see slot() for index)
        self.client_data = [
            {p: {} for p in range(self.player_num)}
            for _ in range(self.list_len)
        ]
        self.begin_time = [None] * self.list_len
        self.end_time   = [None] * self.list_len

        self.fail_playing         = False   # True on error → triggers restart
        self.server_using         = False   # True while a backend-crawl thread runs
        self.round_count          = round_count - 1   # in-round counter
        self.round_count_for_pipe = round_count - 1   # post-round counter

        self.file_create_time = "lobby"      # used in screenshot filenames
        self.mid_pos          = None         # last compare_sim() match center
        self.record_time      = datetime.datetime.now()  # timeout reference
        self.auto_next        = True

    def set_record_time(self, val=None):
        """Reset the timeout reference — call at the start of each test step."""
        self.record_time = datetime.datetime.now()

    def set_attr_by_slot(self, attr, val, round_count):
        temp = getattr(self, attr)
        if isinstance(temp, list):
            raise Exception("Setting wrong attr name in GameState")
        temp[self.slot(round_count)] = val

    def slot(self, round_count):
        """Return the ring-buffer slot index for a given round number."""
        return round_count % self.list_len

# Singleton
instance = _GameState()
def init_game_state(round_count, player_num, list_len) :
    global instance
    instance.final_var(round_count, player_num, list_len)
    return instance

def get_game_state() :
    global instance
    return instance
