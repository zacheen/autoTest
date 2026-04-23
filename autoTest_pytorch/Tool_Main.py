import time
import pyautogui
import cv2
import numpy as np
import sys
import requests
import json
# sys.path.append(".")

import os
from pathlib import Path
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
# import Card

# ----------------------------------------------------------
# Tool main settings
# use_sel : 
    # 0 - using mouse to click (pyautogui) : position is screen position
    # 1 - using selenium to click : position is web page position
use_sel = 1
# ----------------------------------------------------------

Game_envi = None
strech_size = 1
# the time stamp format for test report file name
ISOTIMEFORMAT = '%Y_%m_%d_%H_%M_%S'
format_for_db_time = '%Y-%m-%d %H:%M' 

# checking each folder exist or not, if not, create it
testreport_path = Path("./testreport")
testpic_path = testreport_path / "testpic"
if not testreport_path.exists():
    testreport_path.mkdir()
if not testpic_path.exists():
    testpic_path.mkdir()
print("check/make folder successfully")

# saving the parameters for every game
glo_var = None

class Glo_var():
    # in_game_name : game name, defined in the main file
    # player_num   : max number of players (= how many screenshots to take per round)

    def __init__(self, in_game_name, player_num, round_count, suit_order = None, list_len = 3) :
        # variables initialized only once (do not change when switching games)
        self.game_driver = None  # selenium WebDriver instance (e.g. Chrome)
        base_dir = Path(__file__).resolve().parent
        parent_dir = base_dir.parent
        self.user_change_path = list(parent_dir.rglob("user_change"))[0]

        self.read_input()

        now_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        txt_location_path = testreport_path / now_time
        if not txt_location_path.exists():
            txt_location_path.mkdir()

        self.pipe_output_f = open(txt_location_path / 'pipe_output.txt', "w", encoding='UTF-8') # log for thread
        self.cmd_output_f  = open(txt_location_path / 'cmd_output.txt',  "w", encoding='UTF-8') # log for unittest
        self.error_f       = open(txt_location_path / 'error.txt',        "w", encoding='UTF-8')# log for error
        self.file_create_time = "lobby"

        self.record_time = datetime.datetime.now()
        self.mid_pos = None
        self.auto_next = True

        self.change_by_game(in_game_name, player_num, round_count, suit_order, list_len)
        # self.reset_var(round_count) # Class Glo_var內上方為初始化一次的多個變數，reset_var內為可能會需要"重複初始化"，因此單獨紀錄於一個func內，已便可重複呼叫

    def read_input(self) :
        print("Game_envi :", Game_envi)
        self.is_url = False
        if type(Game_envi)==type("") and len(Game_envi) >= 3 and Game_envi[0:3] == "url" :
            self.is_url = True
            print("Game_envi is url")
            ip = ""
            try :
                response = requests.get("http://"+ip+"/crawler/getCompanys")
                # print("response : "+str(response.content))
                self.DaiLi_data = response.json()
                print("response after json : " + str(self.DaiLi_data))
            except Exception :
                print("json fail so using local file")
                url_json_path = self.user_change_path / 'url.json'
                with open(url_json_path, encoding='UTF-8') as f:
                    self.DaiLi_data = json.load(f)
        else:
            if Game_envi == "Minesweeper_local_py":
                input_file_path = self.user_change_path / "Minesweeper_input.txt"
            else :
                print("Game_envi error (no such env)")
                return

            with open(input_file_path, "r", encoding='UTF-8') as read_input_f:
                self.game_account   = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Account: "     + self.game_account)
                self.game_password  = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Password: "    + self.game_password)
                self.game_agent_ID  = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Agent ID: "    + self.game_agent_ID)
                self.game_money     = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Credits: "     + self.game_money)
                self.game_envir     = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Environment: " + self.game_envir)
                self.server_account  = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Server account: " + self.server_account)
                self.server_password = str(read_input_f.readline().split(" -:")[1]).strip()
                print("Server password: " + self.server_password)

                type_input = ""
                # set type_input = "n" to switch to interactive input mode
                if type_input.strip() == "n" or type_input.strip() == "N" :
                    game_account_in = input("Account (blank = use file): ")
                    if game_account_in.strip() != "":
                        self.game_account = game_account_in
                    game_password_in = input("Password (blank = use file): ")
                    if game_password_in.strip() != "":
                        self.game_password = game_password_in
                    game_agent_ID_in = input("Agent ID (blank = use file): ")
                    if game_agent_ID_in.strip() != "":
                        self.game_agent_ID = game_agent_ID_in
                    game_money_in = input("Credits (blank = use file): ")
                    if game_money_in.strip() != "":
                        self.game_money = game_money_in
                    game_envir_in = input("Environment (blank = use file): ")
                    if game_envir_in.strip() != "":
                        self.game_envir = game_envir_in
                    server_account_in = input("Server account (blank = use file): ")
                    if server_account_in.strip() != "":
                        self.server_account = server_account_in
                    server_password_in = input("Server password (blank = use file): ")
                    if server_password_in.strip() != "":
                        self.server_password = server_password_in
                        
    def change_by_game(self, in_game_name, player_num, round_count, suit_order = None, list_len = 3):
        identify_for_import.game_name = in_game_name

        user_game_pic_parent = self.user_change_path / "game_pic"
        if not user_game_pic_parent.exists():
            user_game_pic_parent.mkdir()

        self.game_pic_path = self.user_change_path / "game_pic" / f"{in_game_name}_pic"
        if not self.game_pic_path.exists():
            self.game_pic_path.mkdir()

        self.cut_pic_path = str(self.game_pic_path / "training_data") + "\\"

        training_data_path = self.game_pic_path / "training_data"
        if not training_data_path.exists():
            training_data_path.mkdir()

        self.game_name = in_game_name
        self.player_num = player_num
        self.list_len = list_len
        try :
            self.class_to_str_list = Data.name_list[in_game_name]
        except KeyError :
            print("Warning: " + in_game_name + " not found in Data.py")
            self.class_to_str_list = {}
        
        # decide card suit order
        # Card.change_suit_order(suit_order)

        self.reset_var(round_count)

    # reset_var holds variables that may need to be re-initialized mid-run (e.g. on error recovery)
    def reset_var(self, round_count) :
        self.client_data = []  # stores all recognition results per round per player
                               # structure: client_data[round % list_len][player_index][key]
        self.begin_time  = []  # round start timestamps (for backend report search)
        self.end_time    = []  # round end timestamps
        for x in range(self.list_len) :  # list_len slots act as a ring buffer (default 3)
            self.client_data.append({})
            for y in range(self.player_num) :
                self.client_data[x][y] = {}
            self.begin_time.append(None)
            self.end_time.append(None)

        self.fail_playing      = False   # set True on error; triggers full restart
        self.server_using      = False   # True while a backend-crawl thread is running
        self.first_time_play   = True
        self.reject_invite     = False
        self.round_count          = round_count - 1  # used during a round
        self.round_count_for_pipe = round_count - 1  # used after a round ends

    # set the timeout reference point; call at the start of each test step
    def set_record_time(self, val = None) :
        if val == None :
            self.record_time = datetime.datetime.now()
        else :
            self.record_time = val

def open_game_web() :
    global glo_var
    print("open browser")
    options = webdriver.ChromeOptions()
    # options.headless = True # A headless browser is a web browser without a graphical user interface (GUI).
    options.add_argument("--window-size=1960,1080")
    options.add_argument('disable-infobars')

    # Configure browser settings to hide the 'controlled by automated software' notification.
    # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.add_experimental_option("useAutomationExtension", False)

    prefs = {"":""}
    prefs["credentials_enable_service"] = False
    prefs["profile.password_manager_enabled"] = False
    options.add_experimental_option("prefs", prefs)
    service = Service(ChromeDriverManager().install())
    glo_var.game_driver = webdriver.Chrome(service=service, options=options)

    pyautogui.click(30, 30) # click the browser window, make it top
    time.sleep(1)
    full_screen()
    # open_book_mark()
    login_plat()

    glo_var.actionChains = ActionChains(glo_var.game_driver)

    # main_windows = glo_var.game_driver.current_window_handle
    # print(main_windows) 
    # all_windows = glo_var.game_driver.window_handles
    # print(all_windows)

def login_plat() :
    global glo_var
    print("login platform")
    if Game_envi == "CQ9" :
        glo_var.game_driver.get("https://h5bt.cqgame.games/h5/BT02/?language=zh-cn&?token=guest")
    elif Game_envi == "Minesweeper_web" :
        glo_var.game_driver.get("http://127.0.0.1:8000")
    else :
        raise Exception(f"Game_envi {Game_envi} doesn't exist!")

def switch_to_game_web():
    # call this after a page opens a new tab, so selenium can control it
    global glo_var
    print("switching web page")
    all_windows = glo_var.game_driver.window_handles
    # < Method 1 >
    # for handle in all_windows:
    #     if handle != main_windows:
    #         driver.switch_to.window(handle)
    # < Method 2 > switch to the last tab
    glo_var.game_driver.switch_to.window(all_windows[-1])

    # pyautogui.click(21, 21)

def full_screen() :
    global glo_var
    # driver.refresh() will disable fullscreen setting
    glo_var.game_driver.maximize_window() # still have the top bar
    # glo_var.game_driver.fullscreen_window() # F11 full screen

    # using hotkey to swtich to fullscreen
    pyautogui.hotkey("f11")


def open_book_mark():
    pyautogui.hotkey("ctrl","shift","b")

# write error info to error.txt
def report_error(round_num, why = None) :
    global glo_var
    glo_var.error_f.write("error round : " + str(round_num) + "\n")
    glo_var.error_f.write("error time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    if why != None :
        glo_var.error_f.write(why + "\n")
    glo_var.error_f.flush()

# read one line from a file handle and return it as a [x, y, w, h] int list
def read_pos(read_dst_f) :
    read_in = read_dst_f.readline().strip().split(", ")
    position = [int(read_in[0]), int(read_in[1]), int(read_in[2]), int(read_in[3])]
    return position

def check_valid_region(pos, region):
    x,y = pos
    for (st_x,st_1,len_n,len_y), in_flag in region :
        if ((st_x <= x <= (st_x+len_n)) and (st_1 <= y <= (st_1+len_y))) != in_flag :
            return False
    return True
    

# click a screen position
# pos          : (x, y) in screen coordinates
# stri         : label to print in the log
# dosleep      : wait time (seconds) after the click
# long_click   : if set, hold the mouse down for this many seconds before releasing
# move_click   : if set, move to the position and wait this many seconds before clicking
# limit_region : if set, only click if pos is inside/outside the defined region
def click(pos, stri = None, dosleep = 0.3, long_click = None, move_click = None, limit_region = None) :
    global glo_var
    global use_sel

    x = pos[0]
    y = pos[1]

    if use_sel == 0 :
        if limit_region != None :
            if not check_valid_region((x,y), limit_region) :
                return False

        if stri != None:
            print_to_output(stri + " click_pos : ("+str(x)+","+str(y)+")")
        else :
            print_to_output("click_pos : ("+str(x)+","+str(y)+")")

        if move_click != None :
            pyautogui.moveTo(x, y)
            time.sleep(move_click)

        if long_click == None :
            pyautogui.click(x, y)
            pyautogui.moveTo(952, 21)
        else :
            pyautogui.mouseDown(x, y)
            time.sleep(long_click)
            pyautogui.mouseUp()
            pyautogui.moveTo(952, 21)

    else :
        if stri != None :
            print_to_output(stri+" click_pos : ("+str(x)+","+str(y)+")")
        else :
            print_to_output("click_pos : ("+str(x)+","+str(y)+")")

        ActionChains(glo_var.game_driver).move_by_offset(x, y).click().move_by_offset(-x, -y).perform()
        dosleep = dosleep - 0.7

    if dosleep > 0:
        time.sleep(dosleep)

    return True

# click the center of the last image found by compare_sim
# also resets the timeout timer — marks the end of the current state and start of the next
def click_mid(stri = "", dosleep = 0.3, long_click = None, move_click = None) :
    global glo_var
    click(glo_var.mid_pos, "click " + stri, dosleep, long_click, move_click)
    glo_var.set_record_time()

# drag the screen left or right (used to scroll the lobby page)
# direction : "left" or "right"
# times     : how many times to drag
# note: url-mode always uses pyautogui — selenium drag causes issues on url-type games
def mouse_drag(direction, times) :
    global glo_var
    if use_sel == 0 or glo_var.is_url:
        if direction == "left" :
            for x in range(times) :
                pyautogui.mouseDown(400, 600)
                pyautogui.moveTo(1500, 600, 1.5)
                time.sleep(0.5)
                pyautogui.mouseUp()
                time.sleep(0.5)
        elif direction == "right" :
            for x in range(times) :
                pyautogui.mouseDown(1500, 600)
                pyautogui.moveTo(400, 600, 1.5)
                time.sleep(0.4)
                pyautogui.mouseUp()
                time.sleep(0.2)
        pyautogui.moveTo(952, 21)
    else :
        if direction == "left":
            for x in range(times) :
                ActionChains(glo_var.game_driver).move_by_offset(400, 600).click_and_hold().perform()
                for y in range(20):
                    ActionChains(glo_var.game_driver).move_by_offset(55, 0).perform()
                ActionChains(glo_var.game_driver).release().perform()
                ActionChains(glo_var.game_driver).move_by_offset(-1500, -600).perform()
        elif direction == "right" :
            for x in range(times) :
                ActionChains(glo_var.game_driver).move_by_offset(1500, 600).click_and_hold().perform()
                for y in range(20):
                    ActionChains(glo_var.game_driver).move_by_offset(-55, 0).perform()
                # print("first stop")
                # time.sleep(1)
                ActionChains(glo_var.game_driver).move_by_offset(0, 0).perform()
                ActionChains(glo_var.game_driver).release().perform()
                ActionChains(glo_var.game_driver).move_by_offset(0, 0).perform()
                # print("second stop")
                # time.sleep(1)
                ActionChains(glo_var.game_driver).move_by_offset(-400, -600).perform()
                # print("third stop")
                # time.sleep(1)

# Unused alternative: Win32 screen capture (faster but unstable — kept for reference)
# import win32gui, win32ui, win32con, win32api
# class Cap_var() :
#     def __init__(self) :
#         hwndDC = win32gui.GetWindowDC(0)
#         self.mfcDC = win32ui.CreateDCFromHandle(hwndDC)
#         self.saveDC = self.mfcDC.CreateCompatibleDC()
#         self.saveBitMap = win32ui.CreateBitmap()
# cap_var = Cap_var()
# def window_capture(filename, region = (0,0,1919,1079)) :
#     global cap_var
#     w, h = region[2], region[3]
#     cap_var.saveBitMap.CreateCompatibleBitmap(cap_var.mfcDC, w, h)
#     cap_var.saveDC.SelectObject(cap_var.saveBitMap)
#     cap_var.saveDC.BitBlt((0, 0), (w, h), cap_var.mfcDC, (region[0], region[1]), win32con.SRCCOPY)
#     cap_var.saveBitMap.SaveBitmapFile(cap_var.saveDC, filename)

# Replacement for pyautogui.locateCenterOnScreen
# (PyAutoGUI 0.9.54 is not compatible with OpenCV 4.11)
def read_template(pic_file) :
    pic_file = str(pic_file)
    return cv2.imread(pic_file)

def locateCenterOnScreen(template_pic, region = None, save_loc = None):
    # Take screenshot
    if region is not None:
        region = tuple(map(int, region))
        
    screenshot = pyautogui.screenshot(region=region)
    
    if save_loc != None:
        screenshot.save(save_loc)
        
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Load and Match template
    # cv2.compareHist(img_cut, img, 0) # This is another method to match template
    result = cv2.matchTemplate(screenshot, template_pic, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    # Calculate center
    h, w = template_pic.shape[:2]
    center_x = max_loc[0] + w // 2
    center_y = max_loc[1] + h // 2
    if region is not None:
        center_x += region[0]
        center_y += region[1]
    return max_val, (center_x, center_y)


def compare_sim(file_place, className, confidence = 0.9, precise = False, before = False, lobby = False) :
    '''
    # Main logic for image comparison and state detection:
    # 1. Compare the current screen with pre-saved images. If similarity > confidence (default 0.9), return the similarity score.
    # 2. If similarity < confidence, check for pixel shifts or offset. If found, return 0.9.
    # 3. If no match is found after both steps, return 0.
    # 
    # Parameters:
    # - file_place: Target file path; pass the filename only (e.g., "continue").
    # - className: Filename for the image to be embedded in the HTML report.
    # - confidence: Threshold for similarity; returns score only if sim > confidence.
    # - precise: If True, skips step 2 (offset check). Use True for fixed-position elements.
    # - before: Screenshot timing. True: Before comparison; False: After finding target; None: Skip screenshot.
    #
    # Optimization Note:
    # There is a trade-off here: taking screenshots at every comparison ensures we have the "last state" for 
    # reports during a crash, but it is time-consuming. Should we move screenshot logic solely to the error handler?
    # Also, screenshots often coincide with timeout checks, but due to latency, the captured image 
    # might not reflect the exact moment of failure (it usually only captures the desired state in normal runs).
    # 
    # Source: Images are pulled from the lobby directory: 
    # C:\Thomas_test\models\research\object_detection\game_pic\lobby_pic
    #
    # Logic Matrix:
    # - confidence > 0.9 & precise = True  => Strict match at a specific location.
    # - confidence > 0.9 & precise = False => Strict match at location; if failed, perform a full-screen search.
    # - confidence <= 0.9 & precise = True  => Relaxed match at location; do not attempt full-screen search.
    '''
    global glo_var

    report_screenshot_path = str(testpic_path / f'{className}_{glo_var.file_create_time}.png')
    # if file_place == "", it means we just want to take screenshot for the report
    if file_place == "" :
        pyautogui.screenshot(report_screenshot_path)
        return None

    if not lobby :
        exact_pos_file = glo_var.game_pic_path / f"{file_place}.txt"
        pic_file       = glo_var.game_pic_path / f"{file_place}.png"
        region_file    = glo_var.game_pic_path / f"{file_place}_region.txt"
    else :
        lobby_path     = glo_var.game_pic_path.parent / "lobby_pic"
        exact_pos_file = lobby_path / f"{file_place}.txt"
        pic_file       = lobby_path / f"{file_place}.png"
        region_file    = lobby_path / f"{file_place}_region.txt"

    template_img     = read_template(pic_file)
    debug_screen_pic = str(testpic_path / f'{glo_var.file_create_time}_{file_place}_detail.png')

    if before == True:
        pyautogui.screenshot(report_screenshot_path)

    sim = 0
    region_sim = 0
    with open(exact_pos_file, "r") as read_dst_f :
        exact_region = read_pos(read_dst_f)
        sim, glo_var.mid_pos = locateCenterOnScreen(template_img, region=exact_region, save_loc=debug_screen_pic)
        print("compare " + file_place + " , sim: " + str(sim))

        if before == False:
            pyautogui.screenshot(report_screenshot_path)

        if sim > confidence :
            return sim

    if confidence <= 0.91 and precise == False:
        find_region = None
        if region_file.exists() :
            with open(region_file, "r") as region_dst_f :
                find_region = read_pos(region_dst_f)

        region_sim, glo_var.mid_pos = locateCenterOnScreen(template_img, region=find_region)
        if before == False :
            pyautogui.screenshot(report_screenshot_path)

        if region_sim > confidence :
            print("< " + file_place + " > not found at exact position, found in full screen")
            return 0.9

    if region_sim < sim :
        print("sim :", sim, "region_sim :", region_sim)
        print("sometimes sim is better than region_sim")
    return max(sim, region_sim)

# Print to console, HTML report, and cmd_output.txt log
def print_to_output(stri) :
    global glo_var
    print(stri)
    HTMLTestRun.p_to_html(str(stri) + "\n")
    glo_var.cmd_output_f.write(str(stri) + "\n")
    glo_var.cmd_output_f.flush()

def cut_pic_data(location, num, round_count, cover = True, cut_new = False, pic_count = None, write_region = False, comp = False):
    '''
    # Captures screenshots of specific regions based on predefined coordinates.
    # 
    # Parameters & Logic:
    # - location: The target item for recognition. Locations and coordinates are manually added via FKNN_pic. 
    #             The folder name is defined by 'location', containing a mandatory 'pos.txt' that stores 
    #             the specific coordinates for cropping. Captured images are saved in their respective 
    #             subfolders under 'user_change'.
    #             Note: The number of folders in the 'location' directory must match the folder count in 'user_change'.
    # - num: The number of coordinate sets (regions) to be read from the position file.
    # - round_count: Syncs the capture process with the current execution round.
    # - cover: 
    #     - False: Appends a timestamp to the filename (typically used for collecting training data).
    #     - True: The image is no longer needed for the training dataset; the file will not be stored 
    #             permanently after the comparison is complete.
    # - cut_new: Currently deprecated.
    # - pic_count: An optional suffix (numeric or string) added to the filename to prevent overwriting 
    #              when capturing multiple images from the same location.
    '''
    
    global glo_var
    end_file_path = glo_var.game_pic_path / location

    png_path = []
    with open(str(end_file_path) + ".txt", "r") as read_dst_f :
        for x in range(num):
            position = read_pos(read_dst_f)

            user_pic_location = glo_var.game_pic_path / location
            if not user_pic_location.exists():
                user_pic_location.mkdir()

            if comp :
                comp_pic_pos = user_pic_location.with_name(user_pic_location.stem + f"_comp_{x+11}_{round_count}")
                png_path.append(str(comp_pic_pos.with_suffix(".png")))
                pyautogui.screenshot(png_path[-1], region=position)
                with open(str(comp_pic_pos.with_suffix(".txt")), "w") as fw :
                    fw.write(str(position)[1:-1])
            elif pic_count == None :
                png_path.append(str(user_pic_location / f"{x+11}_{round_count}.png"))
                pyautogui.screenshot(png_path[-1], region=position)
            else :
                png_path.append(str(user_pic_location / f"{x+11}_{round_count}_{pic_count}.png"))
                pyautogui.screenshot(png_path[-1], region=position)

            if cover == False :
                theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                training_location = Path(glo_var.cut_pic_path) / location
                if not training_location.exists():
                    training_location.mkdir()
                if pic_count == None :
                    pyautogui.screenshot(str(training_location / f"{x}_{theTime}.png"), region=position)
                else :
                    pyautogui.screenshot(str(training_location / f"{pic_count}_{x}_{theTime}.png"), region=position)
    return png_path

# Run OCR/classifier on cut_pic_data screenshots and store results in glo_var.client_data.
# label      : classifier name (matches training_for_XXX and inference_graph_for_XXX)
# name       : key used to store the result — glo_var.client_data[round][player][name]
# use_DATA   : if True, convert classifier output via Data.py label map
# thresh     : classifier confidence threshold (higher = stricter)
def set_client_data(label, name, round_count_in, use_DATA = False, thresh = 0.5, type = "number", class_to_info_list = None, all_in_flag = False) :
    global glo_var
    if type == "number" :
        pass_data = identify_for_import.identify_number(iden_thing=label, round_count=round_count_in, thresh=thresh)
    elif type == "things" :
        pass_data = identify_for_import.identify_things(iden_thing=label, round_count=round_count_in, thresh=thresh, class_to_info_list=class_to_info_list, all_in_flag=all_in_flag)

    for x in range(len(pass_data)) :
        if pass_data[x] == None :
            glo_var.client_data[round_count_in % glo_var.list_len][x][name] = None
        else :
            if use_DATA :
                try :
                    glo_var.client_data[round_count_in % glo_var.list_len][x][name] = glo_var.class_to_str_list[label][pass_data[x]]
                except IndexError :
                    print_to_output(str(name) + " recognition data: " + str(pass_data) + " player " + str(x) + " error")
                    report_error(round_count_in, "recognition error")
            else :
                glo_var.client_data[round_count_in % glo_var.list_len][x][name] = pass_data[x]
            print_to_output("Player " + str(x+1) + " " + name + ": " + str(glo_var.client_data[round_count_in % glo_var.list_len][x][name]))


def can_get_server_data(finish_time, sleep_time = 35) :
    global glo_var

    # ensure at least sleep_time seconds have passed since the round ended
    delta_time = (datetime.datetime.now() - finish_time).seconds
    print("Server wait: " + str(delta_time) + "s")
    if delta_time < sleep_time :
        time.sleep(sleep_time - delta_time)

    if glo_var.server_using :
        total_wait_time = 30
        print("Previous server fetch still running. Waiting " + str(total_wait_time) + "s")
        for x in range(total_wait_time) :
            if x % 10 == 1 :
                print("Server wait remaining: " + str(total_wait_time - x))
            time.sleep(1)
            if glo_var.server_using == False :
                print("Previous fetch done. Starting crawl.")
                break

    if glo_var.server_using :
        print("Server still busy after waiting. Giving up.")
        glo_var.fail_playing = True
        return False

    return True


# Returns True if time since last set_record_time() call exceeds limit seconds
def cal_time_out(limit, state = "") :
    global glo_var
    now_time = datetime.datetime.now()
    delta_time = (now_time - glo_var.record_time).seconds
    # print(delta_time)
    if delta_time >= limit :
        print_to_output(str(state) + " has past " + str(delta_time)+ " sec")
        return True
    else :    
        return False
 
def print_exception(exceptio):
    error_class = exceptio.__class__.__name__
    detail = exceptio.args[0]
    cl, exc, tb = sys.exc_info()
    lastCallStack = traceback.extract_tb(tb)[-1]
    fileName = lastCallStack[0]
    lineNum = lastCallStack[1]
    funcName = lastCallStack[2]
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)
    