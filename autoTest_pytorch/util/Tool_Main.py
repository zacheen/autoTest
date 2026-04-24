"""Tool_Main — test-harness utilities for grid-based automated game testing.

Module layout
─────────────
1. Module constants / paths      — set once on import
2. ClickAbstract strategy         — pyautogui vs selenium, picked by buildClick()
3. Pure utilities                — stateless helpers (read_pos, template matching, ...)
4. GameConfig / GameState — three data containers
5. Glo_var                       — thin composition of the above; module-level singleton
6. Orchestration functions       — compare_sim, cut_pic_data, etc.

Design notes
────────────
• Glo_var holds data, not behaviour. Access is direct — no getters/setters.
• Functions that only need one or two pieces of state take them as explicit params
  (e.g. report_error(error_f, ...), mouse_drag(backend, ...)). Orchestrators that
  touch multiple fields take the whole glo_var.
• Pyautogui vs selenium switching is handled once, inside ClickAbstract subclasses.
  No caller needs to branch on use_sel / is_url directly.
"""

import time
import pyautogui
import cv2
import numpy as np
from pathlib import Path
import datetime

from .Log import print_to_output, report_error, report_screenshot_path, Logger

import util.identify_for_import as identify_for_import
import util.Data as Data

from .Click import ClickPyautogui, get_ctrl
from .Chrome_Driver import Chrome_Driver

# ════════════════════════════════════════════════════════════════════
# 1. Module constants / paths
# ════════════════════════════════════════════════════════════════════

# use_sel: 0 = pyautogui (screen coords) / 1 = selenium (page coords)
use_sel = 1

glo_var = None
def set_glo_var(in_glo_var):
    # Module-level singleton (assigned by the entry script after construction)
    global glo_var
    glo_var = in_glo_var

def open_game_web():
    global _ctrl
    driver = Chrome_Driver(glo_var.env.game_env).driver
    _ctrl = get_ctrl(driver)
    return driver

# ════════════════════════════════════════════════════════════════════
# Pure utility functions
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

# ── input (delegates to backend) ──────────────────────────────────
_ctrl = ClickPyautogui
def click(pos, stri=None, dosleep=0.3,
          long_click=None, move_click=None, limit_region=None):
    """Click at screen/page position pos via the given backend.

    • long_click / move_click are pyautogui-only; ClickSelenium ignores them.
    • limit_region, if given, skips the click when pos falls outside the region
      (returns False instead of clicking).
    • log — optional logger callable; defaults to builtin print.
    """
    x, y = pos
    if limit_region is not None and not check_valid_region((x, y), limit_region):
        return False

    label = (f"{stri} click_pos : ({x},{y})" if stri
             else f"click_pos : ({x},{y})")
    print_to_output(label)

    _ctrl.click(x, y, long_click=long_click, move_click=move_click)

    dosleep += _ctrl.sleep_adjust
    if dosleep > 0:
        time.sleep(dosleep)
    return True

def click_mid(glo_var, stri="", dosleep=0.3, long_click=None, move_click=None):
    """Click the center of the last image found by compare_sim.

    Also resets the timeout timer — marks end-of-state and start-of-next-state.
    """
    click(
        glo_var.state.mid_pos,
        stri="click " + stri,
        dosleep=dosleep, long_click=long_click, move_click=move_click,
    )
    glo_var.state.set_record_time()

def cal_time_out(glo_var, limit, state_name=""):
    """Return True if time since state.record_time exceeds limit seconds."""
    delta = (datetime.datetime.now() - glo_var.state.record_time).seconds
    if delta >= limit:
        print_to_output(f"{state_name} has past {delta} sec")
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
    report_path = report_screenshot_path(className, glo_var.state.file_create_time)

    # file_place == "" means "just take a screenshot for the report"
    if file_place == "":
        pyautogui.screenshot(report_path)
        return None

    pos_file, pic_file, region_file = _resolve_template_paths(
        file_place, glo_var.env.game_pic_path, lobby)

    template  = read_template(pic_file)
    debug_pic = str(Logger.TEST_PIC_PATH / f'{glo_var.state.file_create_time}_{file_place}_detail.png')

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
    game_pic_path = glo_var.env.game_pic_path
    cut_pic_path  = glo_var.env.cut_pic_path
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

from util.Log import get_now_time
def _save_training_shot(region, cut_pic_path, location, idx, pic_count):
    """Save a timestamped copy under training_data/<location>/ for later training."""
    timestamp = get_now_time()
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
    config  = glo_var.env
    slot    = state.slot(round_count_in)

    for x, value in enumerate(results):
        if value is None:
            state.client_data[slot][x][name] = None
            continue

        if use_DATA:
            try:
                state.client_data[slot][x][name] = config.class_to_str_list[label][value]
            except IndexError:
                print_to_output(f"{name} recognition data: {results} player {x} error")
                report_error(round_count_in, "recognition error")
        else:
            state.client_data[slot][x][name] = value

        print_to_output(f"Player {x+1} {name}: {state.client_data[slot][x][name]}")


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
