"""Tool_Main — test-harness utilities for grid-based automated game testing.

Module layout
─────────────
1. Module constants / paths      — set once on import
2. InputBackend strategy         — pyautogui vs selenium, picked by make_backend()
3. Pure utilities                — stateless helpers (read_pos, template matching, ...)
4. GameConfig / GameState / GameSession — three data containers
5. Glo_var                       — thin composition of the above; module-level singleton
6. Orchestration functions       — compare_sim, cut_pic_data, etc.

Design notes
────────────
• Glo_var holds data, not behaviour. Access is direct — no getters/setters.
• Functions that only need one or two pieces of state take them as explicit params
  (e.g. report_error(error_f, ...), mouse_drag(backend, ...)). Orchestrators that
  touch multiple fields take the whole glo_var.
• Pyautogui vs selenium switching is handled once, inside InputBackend subclasses.
  No caller needs to branch on use_sel / is_url directly.
"""

import time
import pyautogui
import cv2
import numpy as np
import sys
import requests
import json
import os
from pathlib import Path
import io
import glob
import warnings
import datetime
import traceback

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import HTMLTestRun
import identify_for_import
import Data


# ════════════════════════════════════════════════════════════════════
# 1. Module constants / paths
# ════════════════════════════════════════════════════════════════════

ISOTIMEFORMAT      = '%Y_%m_%d_%H_%M_%S'   # used in filenames
format_for_db_time = '%Y-%m-%d %H:%M'      # used for DB search timestamps

# where the cursor parks after each click (top-middle of screen)
HOME_POS = (952, 21)

# use_sel: 0 = pyautogui (screen coords) / 1 = selenium (page coords)
# set externally before Glo_var() if you want to override the default
use_sel = 1

# Game_envi: set by the entry script before Glo_var() is constructed
Game_envi = None

# Report output folders (created on import)
testreport_path = Path("./testreport")
testpic_path    = testreport_path / "testpic"
testreport_path.mkdir(parents=True, exist_ok=True)
testpic_path.mkdir(parents=True, exist_ok=True)
print("check/make folder successfully")


# ════════════════════════════════════════════════════════════════════
# 2. InputBackend — strategy for pyautogui vs selenium
# ════════════════════════════════════════════════════════════════════

class InputBackend:
    """Abstract pointer-input backend. Subclasses: PyautoguiBackend, SeleniumBackend."""
    # Subclasses override to shave off sleep that ActionChains already spends
    sleep_adjust = 0.0

    def click(self, x, y, long_click=None, move_click=None):
        raise NotImplementedError

    def drag_swipe(self, direction, times):
        raise NotImplementedError

    def cancel_swipe_hint(self):
        raise NotImplementedError

    def move_home(self):
        """Park the cursor somewhere neutral. Optional (selenium has no real cursor)."""
        pass


class PyautoguiBackend(InputBackend):
    """Screen-coordinate input via pyautogui — works with any window."""
    def click(self, x, y, long_click=None, move_click=None):
        if move_click is not None:
            pyautogui.moveTo(x, y)
            time.sleep(move_click)
        if long_click is None:
            pyautogui.click(x, y)
        else:
            pyautogui.mouseDown(x, y)
            time.sleep(long_click)
            pyautogui.mouseUp()
        self.move_home()

    def drag_swipe(self, direction, times):
        for _ in range(times):
            if direction == "left":
                pyautogui.mouseDown(400, 600)
                pyautogui.moveTo(1500, 600, 1.5)
                time.sleep(0.5)
                pyautogui.mouseUp()
                time.sleep(0.5)
            elif direction == "right":
                pyautogui.mouseDown(1500, 600)
                pyautogui.moveTo(400, 600, 1.5)
                time.sleep(0.4)
                pyautogui.mouseUp()
                time.sleep(0.2)
        self.move_home()

    def cancel_swipe_hint(self):
        # swipe left then back → hint dismissed, no game element actually moved
        pyautogui.mouseDown(1500, 600)
        pyautogui.moveTo(400, 600, 1.2)
        pyautogui.moveTo(1500, 600, 1.2)
        time.sleep(0.5)
        pyautogui.mouseUp()
        time.sleep(0.5)
        self.move_home()

    def move_home(self):
        pyautogui.moveTo(*HOME_POS)


class SeleniumBackend(InputBackend):
    """Page-coordinate input via selenium ActionChains.
    Note: long_click / move_click are pyautogui-only — silently ignored here."""

    sleep_adjust = -0.7  # ActionChains already waits inside perform()

    def __init__(self, driver):
        self.driver = driver

    def click(self, x, y, long_click=None, move_click=None):
        (ActionChains(self.driver)
            .move_by_offset(x, y)
            .click()
            .move_by_offset(-x, -y)
            .perform())

    def drag_swipe(self, direction, times):
        for _ in range(times):
            if direction == "left":
                self._hold_and_drag(start=(400, 600), step=(55, 0),
                                    steps=20, reset=(-1500, -600))
            elif direction == "right":
                self._hold_and_drag(start=(1500, 600), step=(-55, 0),
                                    steps=20, reset=(-400, -600),
                                    re_center_around_release=True)

    def cancel_swipe_hint(self):
        ActionChains(self.driver).move_by_offset(1500, 600).click_and_hold().perform()
        for _ in range(20):
            ActionChains(self.driver).move_by_offset(-55, 0).perform()
        for _ in range(20):
            ActionChains(self.driver).move_by_offset(55, 0).perform()
        ActionChains(self.driver).release().perform()
        ActionChains(self.driver).move_by_offset(-1500, -600).perform()

    def _hold_and_drag(self, start, step, steps, reset,
                       re_center_around_release=False):
        ActionChains(self.driver).move_by_offset(*start).click_and_hold().perform()
        for _ in range(steps):
            ActionChains(self.driver).move_by_offset(*step).perform()
        if re_center_around_release:
            ActionChains(self.driver).move_by_offset(0, 0).perform()
        ActionChains(self.driver).release().perform()
        if re_center_around_release:
            ActionChains(self.driver).move_by_offset(0, 0).perform()
        ActionChains(self.driver).move_by_offset(*reset).perform()


def make_backend(use_sel, is_url, driver=None) -> InputBackend:
    """Pick the pointer backend.

    url-mode always uses pyautogui: selenium drag events are unreliable on
    url-type game pages.
    """
    if use_sel == 0 or is_url:
        return PyautoguiBackend()
    return SeleniumBackend(driver)


# ════════════════════════════════════════════════════════════════════
# 3. Pure utility functions (no state)
# ════════════════════════════════════════════════════════════════════

def read_pos(read_dst_f):
    """Read one '[x, y, w, h]' line from a file handle; return int list.

    read_dst_f is a file handle (not a filename) — position advances after
    each call, so multiple read_pos() calls on the same handle walk through
    consecutive lines.
    """
    parts = read_dst_f.readline().strip().split(", ")
    return [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]


def check_valid_region(pos, region):
    """region is a list of ((x, y, w, h), must_be_inside_flag) pairs."""
    x, y = pos
    for (st_x, st_y, len_x, len_y), in_flag in region:
        inside = (st_x <= x <= st_x + len_x) and (st_y <= y <= st_y + len_y)
        if inside != in_flag:
            return False
    return True


def read_template(pic_file):
    return cv2.imread(str(pic_file))


def locateCenterOnScreen(template_pic, region=None, save_loc=None):
    """Take a screenshot and template-match; return (similarity, (center_x, center_y)).

    Replacement for pyautogui.locateCenterOnScreen, which is incompatible with
    OpenCV 4.11 in PyAutoGUI 0.9.54.
    """
    if region is not None:
        region = tuple(map(int, region))

    screenshot = pyautogui.screenshot(region=region)
    if save_loc is not None:
        screenshot.save(save_loc)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(screenshot, template_pic, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    h, w = template_pic.shape[:2]
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    if region is not None:
        cx += region[0]
        cy += region[1]
    return max_val, (cx, cy)


def full_screen(driver):
    driver.maximize_window()
    pyautogui.hotkey("f11")


def open_book_mark():
    pyautogui.hotkey("ctrl", "shift", "b")


def login_plat(driver, game_envi):
    print("login platform")
    if game_envi == "CQ9":
        driver.get("https://h5bt.cqgame.games/h5/BT02/?language=zh-cn&?token=guest")
    elif game_envi == "Minesweeper_web":
        driver.get("http://127.0.0.1:8000")
    else:
        raise Exception(f"Game_envi {game_envi} doesn't exist!")


def switch_to_game_web(driver):
    """Call this after the page opens a new tab, so selenium can control it."""
    print("switching web page")
    driver.switch_to.window(driver.window_handles[-1])


def print_exception(exc):
    _, _, tb = sys.exc_info()
    last = traceback.extract_tb(tb)[-1]
    msg = (f'File "{last[0]}", line {last[1]}, in {last[2]}: '
           f'[{exc.__class__.__name__}] {exc.args[0]}')
    print(msg)


# ════════════════════════════════════════════════════════════════════
# 4. Config / State / Session — three data containers
# ════════════════════════════════════════════════════════════════════

class GameConfig:
    """Static per-session config — paths, credentials, label map.

    Read-only after init. Owns:
      • game_name, player_num, list_len, game_envi, is_url
      • user_change_path, game_pic_path, cut_pic_path
      • credentials (game_account / game_password / ...)
      • class_to_str_list (label → string map from Data.py)
    """

    def __init__(self, in_game_name, player_num, game_envi, list_len=3):
        self.game_name  = in_game_name
        self.player_num = player_num
        self.list_len   = list_len
        self.game_envi  = game_envi

        self.user_change_path = self._find_user_change_dir()
        self._setup_game_paths()

        self.is_url     = False
        self.DaiLi_data = None
        self._load_credentials()

        self.class_to_str_list = self._load_class_to_str_list()

    # ── setup helpers ──────────────────────────────────────────────

    @staticmethod
    def _find_user_change_dir():
        base_dir = Path(__file__).resolve().parent
        return list(base_dir.parent.rglob("user_change"))[0]

    def _setup_game_paths(self):
        (self.user_change_path / "game_pic").mkdir(exist_ok=True)
        self.game_pic_path = self.user_change_path / "game_pic" / f"{self.game_name}_pic"
        self.game_pic_path.mkdir(exist_ok=True)

        training_data_path = self.game_pic_path / "training_data"
        training_data_path.mkdir(exist_ok=True)
        self.cut_pic_path = str(training_data_path) + "\\"

        # export game name to the identification module
        identify_for_import.game_name = self.game_name

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


class GameState:
    """Dynamic runtime state — mutates during play.

    Call reset(round_count) to re-init for a new playthrough (e.g. after an
    error restart). Owns:
      • round counters (round_count, round_count_for_pipe)
      • ring-buffered per-round data (client_data, begin_time, end_time)
      • flags (fail_playing, server_using, first_time_play, reject_invite)
      • transient values (mid_pos, record_time, file_create_time)
    """

    def __init__(self, round_count, player_num, list_len):
        self._player_num = player_num
        self._list_len   = list_len
        self.reset(round_count)

    def reset(self, round_count):
        # ring buffer: one slot per outstanding round (see slot() for index)
        self.client_data = [
            {p: {} for p in range(self._player_num)}
            for _ in range(self._list_len)
        ]
        self.begin_time = [None] * self._list_len
        self.end_time   = [None] * self._list_len

        self.fail_playing         = False   # True on error → triggers restart
        self.server_using         = False   # True while a backend-crawl thread runs
        self.first_time_play      = True
        self.reject_invite        = False
        self.round_count          = round_count - 1   # in-round counter
        self.round_count_for_pipe = round_count - 1   # post-round counter

        self.file_create_time = "lobby"      # used in screenshot filenames
        self.mid_pos          = None         # last compare_sim() match center
        self.record_time      = datetime.datetime.now()  # timeout reference
        self.auto_next        = True

    def set_record_time(self, val=None):
        """Reset the timeout reference — call at the start of each test step."""
        self.record_time = val if val is not None else datetime.datetime.now()

    def slot(self, round_count):
        """Return the ring-buffer slot index for a given round number."""
        return round_count % self._list_len


class GameSession:
    """I/O resources held open for the life of one playing session.

    Owns:
      • log file handles (pipe_output_f, cmd_output_f, error_f)
      • selenium WebDriver (game_driver)
      • input backend (backend) — chosen by make_backend()
      • actionChains (for direct selenium use outside the backend abstraction)
    """

    def __init__(self, testreport_root):
        now_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        log_dir = testreport_root / now_time
        log_dir.mkdir(parents=True, exist_ok=True)

        self.pipe_output_f = open(log_dir / 'pipe_output.txt', "w", encoding='UTF-8')
        self.cmd_output_f  = open(log_dir / 'cmd_output.txt',  "w", encoding='UTF-8')
        self.error_f       = open(log_dir / 'error.txt',       "w", encoding='UTF-8')

        self.game_driver  = None
        self.backend      = None
        self.actionChains = None

    def bind_driver(self, driver, use_sel, is_url):
        """Register a freshly created WebDriver and construct its input backend."""
        self.game_driver = driver
        self.backend     = make_backend(use_sel, is_url, driver)
        self.actionChains = ActionChains(driver)


# ════════════════════════════════════════════════════════════════════
# 5. Glo_var — thin composition of config/state/session
# ════════════════════════════════════════════════════════════════════

class Glo_var:
    """Session-wide data hub.

    Composition over inheritance:
      • config  — static (paths, credentials, labels)
      • state   — dynamic (counters, flags, buffers)
      • session — I/O resources (logs, driver, backend)

    Direct attribute access is fine — no getter/setter logic. Call sites read
    e.g. `glo_var.state.round_count`, `glo_var.session.backend`.
    """

    def __init__(self, in_game_name, player_num, round_count,
                 suit_order=None, list_len=3):
        self.config  = GameConfig(in_game_name, player_num, Game_envi, list_len)
        self.state   = GameState(round_count, player_num, list_len)
        self.session = GameSession(testreport_path)

    # Thin convenience delegates — common enough to keep at the top level
    def reset(self, round_count):
        self.state.reset(round_count)

    def set_record_time(self, val=None):
        self.state.set_record_time(val)


# Module-level singleton (assigned by the entry script after construction)
glo_var = None


# ════════════════════════════════════════════════════════════════════
# 6. Orchestration / IO functions
# ════════════════════════════════════════════════════════════════════

# ── browser lifecycle ─────────────────────────────────────────────

def open_game_web(glo_var):
    """Open browser, log in, and wire the driver into the session."""
    print("open browser")
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1960,1080")
    options.add_argument("disable-infobars")
    prefs = {
        "": "",
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
    }
    options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    pyautogui.click(30, 30)
    time.sleep(1)
    full_screen(driver)
    login_plat(driver, glo_var.config.game_envi)

    glo_var.session.bind_driver(driver, use_sel, glo_var.config.is_url)


# ── input (delegates to backend) ──────────────────────────────────

def click(backend, pos, stri=None, dosleep=0.3,
          long_click=None, move_click=None, limit_region=None, log=None):
    """Click at screen/page position pos via the given backend.

    • long_click / move_click are pyautogui-only; SeleniumBackend ignores them.
    • limit_region, if given, skips the click when pos falls outside the region
      (returns False instead of clicking).
    • log — optional logger callable; defaults to builtin print.
    """
    x, y = pos
    if limit_region is not None and not check_valid_region((x, y), limit_region):
        return False

    label = (f"{stri} click_pos : ({x},{y})" if stri
             else f"click_pos : ({x},{y})")
    (log or print)(label)

    backend.click(x, y, long_click=long_click, move_click=move_click)

    dosleep += backend.sleep_adjust
    if dosleep > 0:
        time.sleep(dosleep)
    return True


def click_mid(glo_var, stri="", dosleep=0.3, long_click=None, move_click=None):
    """Click the center of the last image found by compare_sim.

    Also resets the timeout timer — marks end-of-state and start-of-next-state.
    """
    def _log(s):
        print_to_output(glo_var.session.cmd_output_f, s)

    click(
        glo_var.session.backend,
        glo_var.state.mid_pos,
        stri="click " + stri,
        dosleep=dosleep, long_click=long_click, move_click=move_click,
        log=_log,
    )
    glo_var.state.set_record_time()


def mouse_drag(backend, direction, times):
    """Swipe left or right (used to scroll the lobby)."""
    backend.drag_swipe(direction, times)


def cancel_first_time(backend):
    """Dismiss the first-time swipe hint (swipe out and back)."""
    print("canceling first time sliding hint")
    backend.cancel_swipe_hint()
    print("end canceling hint")


# ── logging / error reporting ─────────────────────────────────────

def print_to_output(cmd_output_f, stri):
    """Print to console, HTML report, and cmd_output.txt."""
    print(stri)
    HTMLTestRun.p_to_html(str(stri) + "\n")
    cmd_output_f.write(str(stri) + "\n")
    cmd_output_f.flush()


def report_error(error_f, round_num, why=None):
    """Append an error entry to error.txt.

    round_num — which round the error occurred in.
    why       — optional reason string.
    """
    error_f.write(f"error round : {round_num}\n")
    error_f.write(f"error time : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if why is not None:
        error_f.write(why + "\n")
    error_f.flush()


def cal_time_out(glo_var, limit, state_name=""):
    """Return True if time since state.record_time exceeds limit seconds."""
    delta = (datetime.datetime.now() - glo_var.state.record_time).seconds
    if delta >= limit:
        print_to_output(glo_var.session.cmd_output_f,
                        f"{state_name} has past {delta} sec")
        return True
    return False


# ── compare_sim + internal helpers ────────────────────────────────

def compare_sim(glo_var, file_place, className,
                confidence=0.9, precise=False, before=False, lobby=False):
    """Compare the screen with a saved template image.

    Returns a similarity score (0–1), or None when file_place == "" (in which
    case only a report screenshot is taken).

    Also updates glo_var.state.mid_pos to the match center — used by the next
    click_mid() call.

    precise=True  : only search the exact saved position (no fallback).
    precise=False : if not found exactly, search the full screen and return
                    0.9 if found.
    lobby=True    : look in lobby_pic/ instead of the game-specific folder.
    """
    report_path = _report_screenshot_path(className, glo_var.state.file_create_time)

    # file_place == "" means "just take a screenshot for the report"
    if file_place == "":
        pyautogui.screenshot(report_path)
        return None

    pos_file, pic_file, region_file = _resolve_template_paths(
        file_place, glo_var.config.game_pic_path, lobby)

    template  = read_template(pic_file)
    debug_pic = str(testpic_path / f'{glo_var.state.file_create_time}_{file_place}_detail.png')

    if before:
        pyautogui.screenshot(report_path)

    # try exact position first
    sim, pos = _match_at_exact_position(template, pos_file, debug_pic)
    print(f"compare {file_place} , sim: {sim}")
    glo_var.state.mid_pos = pos

    if not before:
        pyautogui.screenshot(report_path)

    if sim > confidence:
        return sim

    # fallback: full-screen search
    region_sim = 0
    if confidence <= 0.91 and not precise:
        region_sim, pos = _match_fallback(template, region_file)
        glo_var.state.mid_pos = pos
        if not before:
            pyautogui.screenshot(report_path)
        if region_sim > confidence:
            print(f"< {file_place} > not found at exact position, found in full screen")
            return 0.9

    if region_sim < sim:
        print(f"sim : {sim} region_sim : {region_sim}")
        print("sometimes sim is better than region_sim")
    return max(sim, region_sim)


def _report_screenshot_path(className, file_create_time):
    return str(testpic_path / f'{className}_{file_create_time}.png')


def _resolve_template_paths(file_place, game_pic_path, lobby):
    """Return (pos_txt, pic_png, region_txt) paths for a template."""
    base = game_pic_path.parent / "lobby_pic" if lobby else game_pic_path
    return (
        base / f"{file_place}.txt",
        base / f"{file_place}.png",
        base / f"{file_place}_region.txt",
    )


def _match_at_exact_position(template, pos_file, save_loc=None):
    with open(pos_file, "r") as f:
        region = read_pos(f)
    return locateCenterOnScreen(template, region=region, save_loc=save_loc)


def _match_fallback(template, region_file):
    """Full-screen (or bounded region) search when the exact-position lookup misses."""
    region = None
    if region_file.exists():
        with open(region_file, "r") as f:
            region = read_pos(f)
    return locateCenterOnScreen(template, region=region)


# ── cut_pic_data + internal helpers ───────────────────────────────

def cut_pic_data(glo_var, location, num, round_count,
                 cover=True, cut_new=False, pic_count=None,
                 write_region=False, comp=False):
    """Take screenshots of predefined regions and save them.

    location  : subfolder/file name under game_pic_path (no extension).
    num       : number of regions to read from the .txt file.
    cover     : True = overwrite same file (comparison use);
                False = also save a timestamped copy in training_data/.
    pic_count : optional extra suffix when multiple shots of the same region
                should not overwrite each other.
    comp      : True = save into a '<location>_comp_<idx>_<round>.png' sibling
                file plus a sidecar .txt containing the region coords.
    """
    game_pic_path = glo_var.config.game_pic_path
    cut_pic_path  = glo_var.config.cut_pic_path
    txt_path      = str(game_pic_path / location) + ".txt"
    out_dir       = game_pic_path / location

    regions = _read_regions(txt_path, num)
    out_dir.mkdir(exist_ok=True)

    png_paths = []
    for idx, region in enumerate(regions):
        if comp:
            png = _save_comparison_shot(region, out_dir, idx, round_count)
        elif pic_count is None:
            png = _save_region_shot(region, out_dir / f"{idx+11}_{round_count}.png")
        else:
            png = _save_region_shot(region, out_dir / f"{idx+11}_{round_count}_{pic_count}.png")
        png_paths.append(png)

        if not cover:
            _save_training_shot(region, cut_pic_path, location, idx, pic_count)

    return png_paths


def _read_regions(txt_path, num):
    """Read `num` region specs from a pos-file."""
    regions = []
    with open(txt_path, "r") as f:
        for _ in range(num):
            regions.append(read_pos(f))
    return regions


def _save_region_shot(region, out_path):
    out = str(out_path)
    pyautogui.screenshot(out, region=region)
    return out


def _save_comparison_shot(region, out_dir, idx, round_count):
    """Save into '<out_dir>_comp_<idx+11>_<round_count>.png' (sibling of out_dir)
    plus a .txt sidecar containing the region coords."""
    stem = out_dir.with_name(out_dir.stem + f"_comp_{idx+11}_{round_count}")
    png  = str(stem.with_suffix(".png"))
    pyautogui.screenshot(png, region=region)
    with open(str(stem.with_suffix(".txt")), "w") as fw:
        fw.write(str(region)[1:-1])
    return png


def _save_training_shot(region, cut_pic_path, location, idx, pic_count):
    """Save a timestamped copy under training_data/<location>/ for later training."""
    timestamp = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    training_dir = Path(cut_pic_path) / location
    training_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{idx}_{timestamp}.png" if pic_count is None
            else f"{pic_count}_{idx}_{timestamp}.png")
    pyautogui.screenshot(str(training_dir / name), region=region)


# ── set_client_data: run recognition and store results ────────────

def set_client_data(glo_var, label, name, round_count_in,
                    use_DATA=False, thresh=0.5, type="number",
                    class_to_info_list=None, all_in_flag=False):
    """Run OCR / classifier on cut_pic_data shots and store results in state.

    label          — classifier name (matches training_for_XXX and
                     inference_graph_for_XXX).
    name           — key used to store the result —
                     state.client_data[slot][player][name].
    round_count_in — the round number; used to pick the ring-buffer slot.
    use_DATA       — if True, convert classifier output via class_to_str_list.
    thresh         — classifier confidence threshold (higher = stricter).
    """
    if type == "number":
        results = identify_for_import.identify_number(
            iden_thing=label, round_count=round_count_in, thresh=thresh)
    elif type == "things":
        results = identify_for_import.identify_things(
            iden_thing=label, round_count=round_count_in, thresh=thresh,
            class_to_info_list=class_to_info_list, all_in_flag=all_in_flag)

    state   = glo_var.state
    config  = glo_var.config
    slot    = state.slot(round_count_in)
    log_f   = glo_var.session.cmd_output_f
    error_f = glo_var.session.error_f

    for x, value in enumerate(results):
        if value is None:
            state.client_data[slot][x][name] = None
            continue

        if use_DATA:
            try:
                state.client_data[slot][x][name] = config.class_to_str_list[label][value]
            except IndexError:
                print_to_output(log_f,
                    f"{name} recognition data: {results} player {x} error")
                report_error(error_f, round_count_in, "recognition error")
        else:
            state.client_data[slot][x][name] = value

        print_to_output(log_f,
            f"Player {x+1} {name}: {state.client_data[slot][x][name]}")


# ── can_get_server_data: coordinate with backend-crawl threads ────

def can_get_server_data(glo_var, finish_time, sleep_time=35):
    """Block until it's safe to fetch backend data for this round.

    Waits at least sleep_time seconds since finish_time, then yields to any
    other backend-crawl thread that's still running.  Returns False (and sets
    fail_playing) if the other thread never releases.
    """
    state = glo_var.state

    # ensure at least sleep_time seconds have passed since the round ended
    delta = (datetime.datetime.now() - finish_time).seconds
    print(f"Server wait: {delta}s")
    if delta < sleep_time:
        time.sleep(sleep_time - delta)

    # yield to any in-flight backend-crawl thread
    if state.server_using:
        total = 30
        print(f"Previous server fetch still running. Waiting {total}s")
        for x in range(total):
            if x % 10 == 1:
                print(f"Server wait remaining: {total - x}")
            time.sleep(1)
            if not state.server_using:
                print("Previous fetch done. Starting crawl.")
                break

    if state.server_using:
        print("Server still busy after waiting. Giving up.")
        state.fail_playing = True
        return False
    return True
