import time
import pyautogui
import cv2
import numpy as np
import sys
# #安裝PY3時，會預設環境變數路徑，但自己定義的工具不會放在裡面，因此透過這個方式，將執行檔當前路徑加入至環境變數，使電腦可以獲取路徑，不會找不到檔案
# sys.path.append(".")

import os
##@
#TS使用到的LOG層級，3為有錯誤才印出 (細節待確認)
# 這個要放在 import tensorflow 之前，因為 import tensorflow 就會印出警示訊息了
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import io
import glob
import warnings
import datetime
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains


import HTMLTestRun
import identify_for_import
# import Data
# import Card

# ----------------------------------------------------------
# Tool main settings
# use_sel : 
    # 0 - using mouse to click (pyautogui) : position is screen position
    # 1 - using selenium to click : position is web page position
use_sel = 0
# ----------------------------------------------------------

Game_envi = None
#當用途為辨識的時候，只能設定為1，若目的為多開視窗進行點擊或進行遊戲產生賽果時，可以設定為2(預期可以開多個視窗)
##@
strech_size = 1
# the time stamp format for test report file name
ISOTIMEFORMAT = '%Y_%m_%d_%H_%M_%S'
format_for_db_time = '%Y-%m-%d %H:%M' 
# screen_num is abandoned, since pyautogui can only click on the main screen
screen_num = 1
# screen_num = 0   (-1920 ~ 0)    # 待測試
# screen_num = 1   (0 ~ 1920)
# screen_num = 2   (1920 ~ 3840)    # 待測試

# checking each folder exist or not, if not, create it
if os.path.isdir("./testreport") == False : 
    os.mkdir("./testreport")
    os.mkdir("./testreport/testpic")
elif os.path.isdir("./testreport/testpic") == False : 
    os.mkdir("./testreport/testpic")
print("check/make folder scuessfully")

# 確認過這個一定要放外面，且每次都一定要讀取(但可以不用使用)
# 設定一個全域變數
glo_var = None #儲存在記憶體內 (俗稱創造一個碗)
# 每個遊戲都要用的參數
class Glo_var():
    # in_game_name→遊戲名稱，於FKNN_MAIN內定義
    # player_num→該遊戲的最高遊戲人數
    # round_count用於計算遊戲當下的回合數
    # func後括號內所帶的內容都稱之參數

    def __init__(self, in_game_name, player_num, round_count, suit_order = None, list_len = 3) : 
        # 只需要初始化一次的東西
        # (不會隨著切換遊戲改變的東西)
        self.game_driver = None #定義遊戲中會使用到的Driver變數(EX:chrome→webdriver)，還有一個後台用的driver
        self.user_abs_loc = os.getcwd() + "\\" + "user_change\\" #定義可透過使用者改變的參數資料夾位置

        self.read_input() # 讀取設定檔
        
        # 創建檔案(不變)
        

        now_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        txt_location = os.getcwd() + "\\testreport\\" + now_time
        if os.path.isdir(txt_location) == False : 
            os.mkdir(txt_location)
        self.pipe_output_f = open(txt_location+"\\"+'pipe_output'+'.txt', "w", encoding='UTF-8') #開啟寫入TXT的寫頭，此為thread分支，用以記錄裁圖→辨識→後台→比對，用來輸出想要印出的內容
        self.cmd_output_f = open(txt_location+"\\"+'cmd_output'+'.txt', "w", encoding='UTF-8') #開啟寫入TXT的寫頭，此為主線Main使用，用來輸出想要印出的內容
        self.error_f = open(txt_location+"\\"+'error'+'.txt', "w", encoding='UTF-8') #開啟寫入TXT的寫頭，此為紀錄有定義過error內容
        self.file_create_time = "lobby" #使用於檔名紀錄時間，並確保該變數會是當前使用的值，此時間為開局時間(等同於teserport創建時間→html)
        
        # input (讀取設定檔)(不變) --------------------------------------------------------------------------------------------------------------------------------------------------------

        # 程式中改變的參數(會隨著程式進行改變) (只需要初始化)
        self.record_time = datetime.datetime.now() #獲取現在時間
        self.mid_pos = None # 圖片中心的座標位置
        self.auto_next = True

        self.change_by_game(in_game_name, player_num, round_count, suit_order, list_len)
        # self.reset_var(round_count) # Class Glo_var內上方為初始化一次的多個變數，reset_var內為可能會需要"重複初始化"，因此單獨紀錄於一個func內，已便可重複呼叫

        # input end -----------------------------------------------------------------------------------------

    def read_input(self) :
        # input (讀取設定檔)(不變) --------------------------------------------------------------------------------------------------------------------------------------------------------
        print("Game_envi :",Game_envi)
        self.is_url = False
        if type(Game_envi)==type("") and len(Game_envi) >= 3 and Game_envi[0:3] == "url" :
            self.is_url = True
            print("Game_envi is url")
            
            try :
                response = requests.get("http://"+ip+"/crawler/getCompanys")
                # print("response : "+str(response.content))
                # 這是一個區塊 用來計算結果是否正確
                self.DaiLi_data = response.json()
                print("response after json : " + str(self.DaiLi_data))

            except Exception : 
                print("json fail so using local file")
                with open(self.user_abs_loc + 'url.json', encoding='UTF-8') as f:
                    self.DaiLi_data = json.load(f)
        else:
            if Game_envi == "CQ9":
                read_input_f = open(self.user_abs_loc + "CQ9_input.txt", "r", encoding='UTF-8') #讀取使用者資料進行登入
            else :
                print("Game_envi error (No this envi)")
            self.game_account = str(read_input_f.readline().split(" -:")[1]).strip() #split(" -:") 透過此方式將資料進行分割，strip() 用以移除字符，開頭或結尾的空格與換行，確保該資料無預期外的字串
            print("遊戲帳號 : " + self.game_account)
            self.game_password = str(read_input_f.readline().split(" -:")[1]).strip()
            print("遊戲密碼 : " + self.game_password)
            self.game_agent_ID = str(read_input_f.readline().split(" -:")[1]).strip()
            print("遊戲代理ID : " + self.game_agent_ID)
            self.game_money = str(read_input_f.readline().split(" -:")[1]).strip()
            print("遊戲分數 : " + self.game_money)
            self.game_envir = str(read_input_f.readline().split(" -:")[1]).strip()
            print("遊戲環境 : " + self.game_envir)
            # 後台
            self.server_account = str(read_input_f.readline().split(" -:")[1]).strip()
            print("後台帳號 : " + self.server_account)
            self.server_password = str(read_input_f.readline().split(" -:")[1]).strip()
            print("後台密碼 : " + self.server_password)
            
            type_input = ""
            #此處為開關，若將input打開，使用者可以自行輸入(也就是把下面一行註解關閉)，若想要使用讀取檔案方式(user_change)，則預設為type_input = ""即可
            #type_input = input("請問輸入是否要從設定檔中讀取? (N/n:不要 其他:要) : ")
            if type_input.strip() == "n" or type_input.strip() == "N" : 
                game_account_in = input("請輸入<<遊戲帳號>> 若為空白則從文字檔中讀取 :")
                if game_account_in.strip() != "":
                    self.game_account = game_account_in
                game_password_in = input("請輸入<<遊戲密碼>> 若為空白則從文字檔中讀取 :")
                if game_password_in.strip() != "":
                    self.game_password = game_password_in
                game_agent_ID_in = input("請輸入<<遊戲代理ID>> 若為空白則從文字檔中讀取 :")
                if game_agent_ID_in.strip() != "":
                    self.game_agent_ID = game_agent_ID_in
                game_money_in = input("請輸入<<加分分數>> 若為空白則從文字檔中讀取 :")
                if game_money_in.strip() != "":
                    self.game_money = game_money_in
                game_envir_in = input("請輸入<<遊戲環境>> 若為空白則從文字檔中讀取 :")
                if game_envir_in.strip() != "":
                    self.game_envir = game_envir_in
                
                server_account_in = input("請輸入<<後台帳號>> 若為空白則從文字檔中讀取 :")
                if server_account_in.strip() != "":
                    self.server_account = server_account_in
                server_password_in = input("請輸入<<後台密碼>> 若為空白則從文字檔中讀取 :")
                if server_password_in.strip() != "":
                    self.server_password = server_password_in
    def change_by_game(self, in_game_name, player_num, round_count, suit_order = None, list_len = 3):
        # 遊玩過程中不會改變 但會隨著不同遊戲改變
        
        # 給 辨識的py 傳遞預設值
        identify_for_import.game_name = in_game_name

        # file location(不變)
        self.now_path = os.getcwd() + "\\" #os.getcwd() 獲取當前路徑，後面加入\ →用以定義路徑資料
        self.pic_path = "game_pic\\" + in_game_name + "_pic\\" #做出 "遊戲名稱_pic"
        self.file_absolute_pos = self.now_path + self.pic_path #定義圖片位置或座標位置的資料夾位置
        
        self.user_abs_pic = self.user_abs_loc + self.pic_path

        # 檢查 user_change 底下是否有 此專案的資料夾
        if os.path.isdir(self.user_abs_loc + "game_pic") == False : 
            os.mkdir(self.user_abs_loc + "game_pic")

        if os.path.isdir(self.user_abs_pic[:-1]) == False : 
            os.mkdir(self.user_abs_pic[:-1])

        self.cut_pic_path = self.file_absolute_pos + r"""training_data""" + "\\" ##@ TS打算進行辨識的訓練資料，待確認 #cut_pic_data cover=False 圖片放置地點

        # 檢查 storage 底下是否有 此專案的資料夾
        if os.path.isdir(self.cut_pic_path[:-1]) == False :  #判斷有沒有定義的資料夾
            os.mkdir(self.cut_pic_path[:-1]) #如果沒有就自動生成

        # 參數(使用者設定) (xxx_Main 裡面設定)
        # player_num_in, player_num 這個遊戲通常有幾個玩家(應該要截幾張圖)
        self.game_name = in_game_name
        self.player_num = player_num #列出該遊戲最多可以參與的人數
        self.list_len = list_len #需要開啟幾個分支，此處預設值為3，用以保證2個若有衝突，還有一個buffer
        try :
            self.class_to_str_list = Data.name_list[in_game_name] ##@ 用來辨識回來的label轉成Str資料型態(當前遊戲)
        except KeyError : 
            print("提醒!!!"+in_game_name+"在Data.py中尚未有資料")
            self.class_to_str_list = {}
        
        # 決定花色順序 
        Card.change_suit_order(suit_order)
        self.reset_var(round_count)

    # 把程式中改變的參數 設定為預設值
    def reset_var(self, round_count) :  # Class Glo_var內上方為初始化一次的多個變數，reset_var內為可能會需要"重複初始化"，因此單獨紀錄於一個func內，已便可重複呼叫
        self.client_data = [] # 定義存放主線與支線資料位置(存放所有遊戲過程中產生的結果資料)  #要注意的是 玩家的key 是base 0的 EX:(4個玩家 0-3) 但後台是(1-4)
        self.begin_time = [] #初始化遊戲起始時間(後台輸贏報表搜尋的起始時間)
        self.end_time = [] #初始化遊戲結束時間(後台輸贏報表搜尋的結束時間)
        for x in range(self.list_len) : #創造三個盤子，可定義為三個盤子中，一回合只會使用一個盤子(EX:早餐、午餐、晚餐) (在該回合產生出來的資料都應該被放進client data內)
            self.client_data.append({}) #舉例TOOL_MAIN設定1個thread(list_len=3，3的目的為buffer用，原則上使用2個)，此時這邊執行完成後，client_data = [{},{},{}]
            for y in range(self.player_num) : 
                # passs = self.client_data[x]
                # passs[y] = {}
                self.client_data[x][y] = {} 
                #舉例FKNN_MAIN設定6個玩家(player_num)，此時這邊執行完成後 
                #client_data = [{0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}]

            self.begin_time.append(None) #[None, None, None]
            self.end_time.append(None) #[None, None, None]
        # print(self.client_data)
    
        self.fail_playing = False # 遊戲中有遇到錯誤狀況時，該參數會被設定為True，此時會將程式內所有項目(包參數、driver、遊戲狀態)，此時遊戲回到大廳，全初始化
        self.server_using = False # 確認是否有人正在使用後台，沒有結束
        self.first_time_play = True # 判斷是否為第一次進入遊戲
        self.reject_invite = False # 判斷是否曾經拒絕過邀請函
        self.round_count = round_count-1 # round_count 是遊戲結束之前用的
        self.round_count_for_pipe = round_count-1 # round_count_for_pipe 是遊戲結束之後用的

    # 設定timeout計時起始點
    # 當 val = None 時 會設定為當下時間
    def s_record_time(self, val = None) : 
        if val == None :
            self.record_time = datetime.datetime.now() # 如果時間變數預設值為空值，則取用當前時間使用
            # print("重整timeout")
        else :
            self.record_time = val # 如果時間變數已有定義，則取用當前值使用

# open chrome for game ############################################################################################################################################################################
# 打開遊戲網頁到平台登入頁面
def open_game_web() :
    global glo_var
    webdriver_path = glo_var.user_abs_loc+'chromedriver.exe'
    options = Options()
    # options.headless = True #Headless Browser是没有沒有圖形介面(GUI)的web瀏覽視窗
    options.add_argument("--window-size=1960,1080")
    options.add_argument('disable-infobars')
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    prefs = {"":""}
    prefs["credentials_enable_service"] = False
    prefs["profile.password_manager_enabled"] = False
    options.add_experimental_option("prefs", prefs)
    # 設定瀏覽器設定值為不出現此溜覽器正透過自動化視窗控制
    glo_var.game_driver = webdriver.Chrome(executable_path=webdriver_path, options=options) #透過設定值開啟瀏覽器
    glo_var.game_driver.maximize_window() #全螢幕

    # 記得這邊一定要用 pyautogui.click
    # 網頁置頂
    pyautogui.click(21, 21) #點擊溜覽器視窗頁面，確保置頂
    # 叫出書籤欄
    # pyautogui.hotkey("ctrl","shift","b")
    pyautogui.moveTo(952, 21) #定義為初始位置，避免滑鼠在溜覽器任意位置內出現提示訊息影響辨識或截圖
    time.sleep(1)
    pyautogui.hotkey("f11")

    print(glo_var.game_driver.get_window_size())

    glo_var.actionChains = ActionChains(glo_var.game_driver)

    # 現在有哪些分頁
    # main_windows = glo_var.game_driver.current_window_handle
    # print(main_windows) 
    # all_windows = glo_var.game_driver.window_handles
    # print(all_windows)

# open game to desktop ############################################################################################################################################################################
# 使用 glo_var讀取的資料 登入一部的登入平台
# ?? 大改
def login_plat() :
    global glo_var #將全域變數導入func，以便後續取用
    if Game_envi == "CQ9" :
        # 開啟該網頁連結
        # ???網址
        glo_var.game_driver.get("https://h5bt.cqgame.games/h5/BT02/?language=zh-cn&?token=guest")
        # 要開到遊戲頁面
    else :
        print("Game_envi error")

# 這個是因為有時候登入平台會開新分頁 所以會有這個
# 如果會的話在畫面讀取完成後要執行這個才可以用selenium操控
def switch_to_game_web():
    global glo_var
    print("switching web page")
    all_windows = glo_var.game_driver.window_handles
    # 方法1
    # for handle in all_windows:
    #     if handle != main_windows:
    #         driver.switch_to.window(handle)
    # 方法2
    # 切換到最後一個分頁
    glo_var.game_driver.switch_to.window(all_windows[-1])

    # pyautogui.click(21, 21) #點擊溜覽器視窗頁面，確保置頂

# 記錄錯誤時間 並輸出到 user_change//error.txt
# round_num 用來記錄是哪一回合出錯
# why 是有可能的錯誤原因
def report_error(round_num, why = None) : 
    global glo_var
    glo_var.error_f.write("error round : " + str(round_num) +"\n") #寫入錯誤格式與內容
    glo_var.error_f.write("error time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n") #寫入錯誤格式與內容
    if why != None :
        glo_var.error_f.write(why+"\n") #如果原因不是空值，就將原因記錄進LOG
    glo_var.error_f.flush() #將寫入內容進行一次儲存，當程式還在運作時，已經寫入的error可以重新讀取檔案開啟瀏覽

# 從 read_dst_f 讀取一行 並把此行 圖片位置座標回傳
# read_dst_f : 是讀寫頭 並不是檔名 所以做完讀寫頭的位置是會改變的
def read_pos(read_dst_f) : #將要讀寫的座標進行寫並處理資料，將頭尾的字符去掉，且把逗號分離
    read_in = read_dst_f.readline()
    print("read_in :", read_in)
    read_in = read_in.strip()
    read_in = read_in.split(", ")

    # if screen_num == 0 :
    #     position = [int(read_in[0]) - 1920,int(read_in[1]),int(read_in[2]),int(read_in[3])]   # 待測試
    # elif screen_num == 1:
    position = [int(read_in[0])       ,int(read_in[1]),int(read_in[2]),int(read_in[3])] #透過int型態儲存為list
    # elif screen_num == 2:
    #     position = [int(read_in[0]) + 1920,int(read_in[1]),int(read_in[2]),int(read_in[3])]   # 待測試
    
    return position #回傳資料置回傳資料至read_pos，使用完成後該參數即消失

# 點擊位置 pos EX: (X,Y)
# stri 是點擊時想輸出的字串
# dosleep 點擊完後 要等幾秒 才進續進行
# long_click 帶入數字 代表要點擊XX秒才放開
# move_click 帶入數字 滑鼠會先移動到上面 等待XX秒 後才進行點擊
def click(pos, stri = None, dosleep = 0.3, long_click = None, move_click = None) : 
    #呼叫全域變數
    global glo_var 
    global use_sel 
    
    x = pos[0] # 定義pos內x的位置
    y = pos[1] # 定義pos內y的位置
    
    if use_sel == 0 : #使用pyautogui
        if stri != None :
            print_to_output(stri+" click_pos : ("+str(x)+","+str(y)+")")
        else :
            print_to_output("click_pos : ("+str(x)+","+str(y)+")")
        
        # 拖曳的code
        # pyautogui.mouseDown(x, y) #點下滑數左鍵(不放開)
        # pyautogui.moveTo(x+1000, y+500, 1.5)  #向左拖曳
        # time.sleep(0.5) 
        # pyautogui.mouseUp() #放開滑鼠左鍵

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

    else : #使用selenium
        # y = y - 130

        # x = (x/strech_size/5)*4 #透過比例變化進行
        # y = (y/strech_size/5)*4 #透過比例變化進行

        if stri != None :
            print_to_output(stri+" click_pos : ("+str(x)+","+str(y)+")")
        else :
            print_to_output("click_pos : ("+str(x)+","+str(y)+")")

        ActionChains(glo_var.game_driver).move_by_offset(x, y).click().move_by_offset(-x, -y).perform()

        dosleep = dosleep-0.7

    if dosleep > 0:
        time.sleep(dosleep)

# 點擊最後一張 compare_sim 找到的圖片的位置的正中間
# stri : 是點擊時想輸出的字串
# dosleep 點擊完後 要等幾秒 才進續進行
# long_click 帶入數字 代表要點擊XX秒才放開
# move_click 帶入數字 滑鼠會先移動到上面 等待XX秒 後才進行點擊
def click_mid( stri = "", dosleep = 0.3, long_click = None, move_click = None) : 
    global glo_var
    click(glo_var.mid_pos, "click " + stri, dosleep, long_click, move_click)

    glo_var.s_record_time() # 紀錄最後一個點擊後，用來記錄當前狀態的結束時間，同時也是下一個狀態的起始時間(可透過FKNN有無牛回合進行回憶→點了四次，最後一次為主)

# 在 compare_sim 裡面用的 用來記錄什麼東西 在哪個位置找到 
# mid_pos_file 輸出的檔案位置
# mid_pos : 寫入檔案的圖片位置
# (原本要用來紀錄有可能出現的位置 不過現在這個沒什麼用)
# def write_mid_pos(mid_pos_file, mid_pos) : 
#     with open(mid_pos_file, "w") as mid_pos_file_f : 
#         mid_pos_file_f.write(str(mid_pos)[1:-1])
#         mid_pos_file_f.flush()

# 向左向右滑動
# 點擊不放 移動滑鼠 放開
# (用來滑動大廳頁面)
# direction 滑動方向 
# times 滑動多少次
def mouse_drag(direction, times) : #透過direction進行方向定義
    global glo_var
    if use_sel == 0 or glo_var.is_url: #使用pyautogui  #url的滑動如果用 selenium 會出問題
        if direction == "left" :
            for x in range(times) :
                pyautogui.mouseDown(400, 600) #點下滑數左鍵(不放開)
                pyautogui.moveTo(1500, 600, 1.5)  #向左拖曳
                time.sleep(0.5) 
                pyautogui.mouseUp() #放開滑鼠左鍵
                time.sleep(0.5)
        elif direction == "right" :
            for x in range(times) :
                pyautogui.mouseDown(1500, 600)
                pyautogui.moveTo(400, 600, 1.5) #向右拖曳
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
                # print("第一次sleep")
                # time.sleep(1)
                ActionChains(glo_var.game_driver).move_by_offset(0, 0).perform()
                ActionChains(glo_var.game_driver).release().perform()
                ActionChains(glo_var.game_driver).move_by_offset(0, 0).perform()
                # print("第二次sleep")
                # time.sleep(1)
                ActionChains(glo_var.game_driver).move_by_offset(-400, -600).perform()
                # print("第三次sleep")
                # time.sleep(1)

# 這個動作可以取消掉第一次可以向右滑的提示 就算沒有 也不會動到畫面中任何東西的位置(因為有滑回原點)
def cancel_first_time() : #
    global glo_var
    print("canceling first time sliding hint")
    # 滑掉第一次的 提示 (點下去 向左滑 向右滑(滑回點下去的位置) 放開)
    if use_sel == 0 : #使用pyautogui
        pyautogui.mouseDown(1500, 600)
        pyautogui.moveTo(400, 600, 1.2)
        pyautogui.moveTo(1500, 600, 1.2)
        time.sleep(0.5)
        pyautogui.mouseUp()
        time.sleep(0.5)
        pyautogui.moveTo(952, 21)
    else :
        ActionChains(glo_var.game_driver).move_by_offset(1500, 600).click_and_hold().perform()
        for y in range(20):
            ActionChains(glo_var.game_driver).move_by_offset(-55, 0).perform()
        for y in range(20):
            ActionChains(glo_var.game_driver).move_by_offset(55, 0).perform()
        ActionChains(glo_var.game_driver).release().perform()
        ActionChains(glo_var.game_driver).move_by_offset(-1500, -600).perform()
    print("end canceling hint")

# # 螢幕截圖 用 win內建的 function 速度比較快 但不知道為什麼會出問題
# import win32gui, win32ui, win32con, win32api
# class Cap_var() :
#     def __init__(self) :
#         hwnd = 0 # 視窗的編號，0號表示當前活躍視窗
#         # 根據視窗控制代碼獲取視窗的裝置上下文DC（Divice Context）
#         hwndDC = win32gui.GetWindowDC(hwnd)
#         # 根據視窗的DC獲取mfcDC
#         self.mfcDC = win32ui.CreateDCFromHandle(hwndDC)
#         # mfcDC建立可相容的DC
#         self.saveDC = self.mfcDC.CreateCompatibleDC()
#         # 建立bigmap準備儲存圖片
#         self.saveBitMap = win32ui.CreateBitmap()

# cap_var = Cap_var()
# def window_capture(filename, region = (0,0,1919,1079)) :
#     global cap_var
#     w = region[2]
#     h = region[3]
#     # 為bitmap開闢空間
#     cap_var.saveBitMap.CreateCompatibleBitmap(cap_var.mfcDC, w, h)
#     # 高度saveDC，將截圖儲存到saveBitmap中
#     cap_var.saveDC.SelectObject(cap_var.saveBitMap)
#     # 擷取從左上角（0，0）長寬為（w，h）的圖片
#     cap_var.saveDC.BitBlt((0, 0), (w, h), cap_var.mfcDC, (region[0], region[1]), win32con.SRCCOPY)
#     cap_var.saveBitMap.SaveBitmapFile(cap_var.saveDC, filename)

# 這個是Main在用的
# 1. 與之前儲存的圖片比較相似度 如果圖片相似度 > confidence(預設0.9) 則回傳圖片相似度
# 2. 如果 < confidence 先找是否有移位 如果有找到會回傳 0.9
# 3. 如果都找不到則回傳0
# file_place 檔案位置，帶入名稱為檔名即可("continue")
# className 要寫入html圖片的檔案名稱
# confidence 是 sim 要大於多少才會回傳sim
# precise 若為True : 則不會進2. False反之  (如果東西是固定位置 precise就應該要 = True)
# before 截圖時間點 True: 先截圖再比對  False: 找到之後再截圖 None: 本次比較不截圖
# 這裡會出現一個問題 就是我希望比較的時候截圖 這樣不管在哪裡中斷 都會有中斷時的畫面(在report) 但截圖非常的花時間 所以要不要在錯誤處理時在截圖就好?
# 而且基本上截圖完之後又會馬上檢查是否有timeout 所以可以說是基本上是同時截圖的
# 而且現在還有一個問題是 因為timeout所以 截的圖不一定是我們想要的(通常只有正常的情況才會截到我想要的圖)
# lobby: 會從 C:\Thomas_test\models\research\object_detection\game_pic\lobby_pic 找圖跟位置
# 這裡面會自動截要放到html的圖
# confidence > 0.9  且 precise = True  ==   回傳值要大於某個數字   ==   要在精準的位置"用很嚴格的標準"找到才算有找到
# confidence > 0.9  且 precise = False ==   如果在精準的位置"用很嚴格的標準"(confidence)找不到，還可以用全螢幕找找看
# confidence <= 0.9 且 precise = True  ==   如果在精準的位置"用不嚴格的標準"找不到，也不要在全螢幕找了
def compare_sim(file_place, className, confidence = 0.9, precise = False, before = False, lobby = False) : 
    global glo_var #導入全域變數

    if file_place == "" : #用於遊戲流程進行中，若有想要把過程中發生的狀態進行截圖，可透過此func進行(compare_sim("", className))，此方法目的僅為在該狀態中沒有要進行結圖比對或辨識，僅儲存圖片
        pyautogui.screenshot('./testreport/testpic/'+className+ "_"+ glo_var.file_create_time+'.png')
        #'./testreport/testpic/→當前目錄與指定資料夾
        #+className+ "_"+ glo_var.file_create_time+'.png'→定義截圖檔名
        return None #結束點

    default_dest = True
    #測試前需先進行的前置作業

    if default_dest :
        #有小工具
        pos_file = glo_var.file_absolute_pos + file_place + ".txt" # FKNN_pic內的座標位置檔案，是自己先創建確認好的，為比對位置標準
        pic_file = glo_var.file_absolute_pos + file_place + ".PNG" # FKNN_pic內的圖片檔案(EX:搶一倍，繼續遊戲)，是自己先創建確認好的，可以拿來判斷狀態，也可以拿來進行點擊
        region_file = glo_var.file_absolute_pos + file_place + "_region.txt"
        # 當pos_file比對失敗時，可透過定義一個範圍，重新再找一次(若沒有此檔案，預設為找全部畫面，但避免找到類似的發生誤導，因此可限定區域)，此參數也是由使用者決定是否提供
        # 如果是要找共同的圖片 就把位置改成lobby 圖都放在這底下
        if lobby :
            pos_file = pos_file.replace(glo_var.game_name, "lobby")
            pic_file = pic_file.replace(glo_var.game_name, "lobby")
            region_file = region_file.replace(glo_var.game_name, "lobby")
        

    # 開啟以前圖片
    img = cv2.imread(pic_file) #讀取原始標準圖

    save_loc = './testreport/testpic/'+ glo_var.file_create_time + "_" + file_place +'_detail.png' #實際測試過程在指定區域的截圖照片(被比對的圖要儲存的位置)

    if before == True: #在下一個狀態要進行動作之前先進行截圖(else:在做完動作後再進行截圖)→兩者差異為可能因為時間差間接影響實際截圖出來的結果，可透過實際執行進行驗證
        pyautogui.screenshot('./testreport/testpic/'+className+ "_"+ glo_var.file_create_time+'.png')

    with open(pos_file, "r") as read_dst_f : 
        position = read_pos(read_dst_f) #根據提供的座標(pos_file)
        # 不管怎樣都要換成現在比的東西的位置 (因為有些東西可能是判斷不一樣 反而要在這個位置做動作)
        glo_var.mid_pos = [position[0]+(position[2]/2),position[1]+(position[3]/2)] # 計算圖片中心點

        pyautogui.screenshot(save_loc, region=position) #讀取要截圖的座標位置(region)，並儲存在指定的資料夾路徑內(save_loc)
        
        img_cut = cv2.imread(save_loc) #將上一動截圖的檔案讀取出來

        # 比較相似度
        # sim = cv2.compareHist(img_cut, img, 0)
        sim = cv2.matchTemplate(img_cut, img, cv2.TM_CCOEFF_NORMED)[0][0] ##@ 運作邏輯與細節待確認 將正確的圖片img與指定截取的圖片img_cut進行比對
        print("比較"+file_place+" , sim : "+str(sim))

        if before == False: #else:在做完動作後再進行截圖
            pyautogui.screenshot('./testreport/testpic/'+className+ "_"+ glo_var.file_create_time+'.png')

        # print(sim)
        # try :
        #     sim = sim[1][1]
        # except :
        #     try : 
        #         sim = sim[0][0]
        #     except :
        #         pass
        if confidence > 0.91 : #這邊為可手動設定confidence(信心度或嚴格度)，預設為需大於0.91，可根據測試需求指定更高層級(EX:0.97)
            # print(file_place + " sim: " + str(sim))
            if sim > confidence : #判斷sim比對結果是否大於手動設定的confidence
                # write_mid_pos(mid_pos_file,glo_var.mid_pos)
                return sim #如果是則回傳sim比較值

        elif sim > 0.91 : #若執行到這一行，進入這個條件(confidence設定<0.91)，表示confidence在這邊無效用
            # write_mid_pos(mid_pos_file,glo_var.mid_pos)
            return sim

    if precise == False: #當confidence設定大於0.91且當sim小於confidence時，或是sim小於預設的0.91，兩個條件其中一個成立時執行這段
        #如果比對找不到 就從畫面找
        # locateCenterOnScreen 原理是用 cv2.matchTemplate
        region = None #將region創造出來
        if os.path.isfile(region_file) : # 確認region_file檔案是否存在(代表是否要用region)
            with open(region_file, "r") as region_dst_f : 
                region = read_pos(region_dst_f)

        if region == None :
            pic_position = pyautogui.locateCenterOnScreen(pic_file, grayscale=False, confidence = 0.93) #透過全畫面找尋是否有pic_file這個檔案圖片
        else :
            pic_position = pyautogui.locateCenterOnScreen(pic_file, grayscale=False, confidence = 0.93, region = region) #透過定義過的region找尋是否有pic_file這個檔案圖片

        if before == False : #若第二次判斷沒有return，則會進行此截圖，複寫第二次的結果
            pyautogui.screenshot('./testreport/testpic/'+className+ "_"+ glo_var.file_create_time+'.png')

        if pic_position != None :
            print("< " + file_place + " > cannot find at particular position") #意味著如果透過上一動的方式可以找到座標，那就是畫面中有能找到這個座標位置
            # 這裡如果找到東西 也會更新位置 但是是找到東西的位置
            glo_var.mid_pos = pic_position
            # write_mid_pos(mid_pos_file,glo_var.mid_pos)
            # 0.9 這是我自己定義的
            return 0.9 #因此回傳0.9，但表示圖片有位移，透過0.9表示

    return 0 #如果手動設定的confidence與預設的0.91及最後的位移判斷()0.9)都失效了，就意味著找不到，回傳0

# 一次比較多個圖片與其對應到的位置 若某個圖片對應的位置 相似度高就會回傳
# 但 它又扣到 是假設 pos 一定很多個
# def compare_sim_groupe(file_place,x,read_dst_f, return_sim = False) :
#     global mid_pos

#     pic_file = file_absolute_pos + file_place + "\\" + str(x) + ".PNG"

#     position = read_pos(read_dst_f)
#     # print(position)
#     window_capture(file_absolute_pos + "pass.PNG", region=position)
    
#     img_cut = cv2.imread(file_absolute_pos + "pass.PNG")
#     img_cut = cv2.calcHist([img_cut], [0], None, [256], [0, 256])
#     img_cut = cv2.normalize(img_cut, img_cut, 0, 1, cv2.NORM_MINMAX, -1)
    
#     img     = cv2.imread(pic_file)
#     img     = cv2.calcHist([img], [0], None, [256], [0, 256]) 
#     img     = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, -1) 

#     sim = cv2.compareHist(img_cut, img, 0)

#     if return_sim :
#         return sim

#     # 不一樣代表是數字
#     if sim < 0.97 :
#         mid_pos = [position[0]+(position[2]/2),position[1]+(position[3]/2)]
#         return position

#     return None

# def just_compare_sim(position, file_place):
#     global glo_var
    
#     if os.path.isfile(file_place) : 
#         # print("file_place : "+str(file_place))
#         pic_file = file_place

#         window_capture(glo_var.file_absolute_pos + "pass.PNG", region=position)
        
#         img_cut = cv2.imread(glo_var.file_absolute_pos + "pass.PNG")
#         img_cut = cv2.calcHist([img_cut], [0], None, [256], [0, 256])
#         img_cut = cv2.normalize(img_cut, img_cut, 0, 1, cv2.NORM_MINMAX, -1)
        
#         img     = cv2.imread(pic_file)
#         img     = cv2.calcHist([img], [0], None, [256], [0, 256])
#         img     = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, -1) 

#         sim = cv2.compareHist(img_cut, img, 0)

#         return sim
#     else :
#         return -1
    
# # 在某個資料夾中找出與此圖最相似的圖片並回傳
# def find_most_sim(file_place, position, num = None) : 
#     # 把全部符合 file_place + _ + x 的檔案抓出來 回傳最高的 similar
#     max_sim = 0
#     max_num = -1

#     if num == None : 
#         targetPattern = file_place+ "_*.PNG"
#         all_file = glob.glob(targetPattern)
#         for this_png in all_file :
#             pass_sim = just_compare_sim(position, this_png)
#             # print(this_png + " 的 sim : " + str(pass_sim))
#             if pass_sim > max_sim :
#                 max_sim = pass_sim
#         return (max_sim, this_png)

#     else :
#         for x in range(num+1):
#             pass_sim = just_compare_sim(position, file_place + "_" + str(x)+ ".PNG")
#             # print(str(x)+" 的 sim : "+str(pass_sim))
#             if pass_sim > max_sim :
#                 max_sim = pass_sim
#                 max_num = x
#         return (max_sim, max_num)

# 輸出字串到3個位置
# 1. cmd 2. html 3. user_change//cmd_output.txt 
def print_to_output(stri) : 
    global glo_var
    print(stri)
    HTMLTestRun.p_to_html(str(stri) + "\n") #印html內
    glo_var.cmd_output_f.write(str(stri) + "\n")
    glo_var.cmd_output_f.flush()

# 透過讀取的位置截圖相應的位置
# location 先透過FKNN_pic手動新增要辨識的項目位置與座標，資料夾檔名為location(自己命名)，裡面的txt檔名固定為pos.txt，裡面儲存的座標位置為要截圖的位置，截圖完成後會儲存在user_change的各自location資料夾內
# location資料夾數量應該要與user_change內的資料夾數量一致
# num 要讀取圖片位置的檔案裡面有幾個位置(有幾組座標就是會是多少)
# round_count 跟 現在第幾回合有關
# cover False : 會在檔名後面加上時間 (通常用來截要用來training的圖片) True : 表示已經不需要加入訓練資料中，因此當下截圖比對完成後就不另外儲存
# cut_new 目前廢棄使用
# pic_count 如果同一個位置要截很多張圖片 但又不想被覆蓋就加上數字(其實也可以是文字)
def cut_pic_data(location, num, round_count, cover = True, cut_new = False, pic_count = None):
    # location 格式 fin_card_num\\
    global glo_var
    # print("BBBB glo_var.file_absolute_pos : ",glo_var.file_absolute_pos)
    end_file_dst = glo_var.file_absolute_pos + location
    print("end_file_dst : "+end_file_dst + ".txt")

    with open(end_file_dst + ".txt", "r") as read_dst_f :
        for x in range(num):
            position = read_pos(read_dst_f)
            if os.path.isdir(glo_var.user_abs_pic + location) == False :  #判斷有沒有定義的資料夾
                os.mkdir(glo_var.user_abs_pic + location) #如果沒有就自動生成
            if pic_count == None :
                pyautogui.screenshot(glo_var.user_abs_pic + location + str(x+11)+"_"+str(round_count)+".PNG", region=position) #透過已定義的座標位置進行截圖
            else :
                pyautogui.screenshot(glo_var.user_abs_pic + location + str(x+11)+"_"+str(round_count)+"_"+str(pic_count)+".PNG", region=position) #透過已定義的座標位置進行截圖
            # 用來切特定位置 不一樣的圖
            # 有 _r 跟 _b 目前好像只能用在牌上?? 所以先注解掉
            # if cut_new == True :
            #     # 先判斷有無相似的圖案
            #     sim, num = find_most_sim(end_file_dst+str(x)+"_r", position)
            #     sim2, num2 = find_most_sim(end_file_dst+str(x)+"_b", position)
            #     if sim < 0.99 and sim2 < 0.99: 
            #         #儲存
            #         theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT) 
            #         pyautogui.screenshot(glo_var.user_abs_loc + location + str(x)+"_"+theTime+".PNG", region=position)
            
            if cover == False : 
                theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                if os.path.isdir(glo_var.cut_pic_path + location) == False :  #判斷有沒有定義的資料夾
                    os.mkdir(glo_var.cut_pic_path + location) #如果沒有就自動生成
                
                if pic_count == None :
                    pyautogui.screenshot(glo_var.cut_pic_path + location + "//" + str(x)+"_"+theTime+".PNG", region=position) #截圖(檔名多了時間)
                else :
                    pyautogui.screenshot(glo_var.cut_pic_path + location + "//" + str(pic_count)+"_"+str(x)+"_"+theTime+".PNG", region=position) #透過已定義的座標位置進行截圖

# 把 cut_pic_data 截好的圖片 辨識後 放入 glo_var.client_data 中
# 這個 funciton 只能用<每個玩家>都<只有一個>的<數字>資料
# label : 要使用哪個訓練好的辨識神經 (會抓取 inference_graph_for_XXX 和 training_for_XXX 的設定)
# 通常會取跟 cut_pic_data 帶入 location 的名稱相同，但是要去掉尾端的\
# name : 取值時候的key  EX: glo_var.client_data[round_count(第幾回合)][第幾個玩家][name]
# round_count_in : 取值時候的key  EX: glo_var.client_data[round_count_in][第幾個玩家][要取的東西]
# use_DATA 如果 辨識結果 有要轉 換成 資料 要打開
#    會讀 Data.py 轉成字串
# thresh 辨識神經 所使用的域值 0~1 愈接近1愈嚴謹、嚴格 == 回傳的資料愈少
# (以下尚未實做)
# 如果有很多個，還沒實做
# 如果只有一個 EX:局號 判斷完直接寫入即可
def set_client_data(label, name, round_count_in, use_DATA = False, thresh = 0.5, type = "number", class_to_info_list = None, all_in_flag = False) : 
    global glo_var
    # 辨識結果
    # print("輸入:", label, round_count_in, thresh)
    if type == "number" :
        pass_data = identify_for_import.identify_number(iden_thing = label, round_count = round_count_in, thresh = thresh) #辨識
        
    elif type == "things" :
        pass_data = identify_for_import.identify_things(iden_thing = label, round_count = round_count_in, thresh = thresh, class_to_info_list = class_to_info_list, all_in_flag = all_in_flag) #辨識
    
    # for 每個玩家
    for x in range(len(pass_data)) :
        # print("in set_client_data x = "+str(x))
        # 如果回傳回來是空的
        if pass_data[x] == None :
            glo_var.client_data[round_count_in % glo_var.list_len][x][name] = None
        # 如果有結果
        else :
            # 看需不需要再用 Data.py 轉成資料
            # 需要
            if use_DATA :
                try :
                    glo_var.client_data[round_count_in % glo_var.list_len][x][name] = glo_var.class_to_str_list[label][pass_data[x]]
                except IndexError :
                    print_to_output(str(name) + " 辨識完資料 : " + str(pass_data) + "第" + str(x) + "個 出錯")
                    report_error(round_count_in , "辨識錯誤")
            # 不需要
            else :
                glo_var.client_data[round_count_in % glo_var.list_len][x][name] = pass_data[x]
            print_to_output("辨識 玩家 第" + str(x+1) +"個 " + name + " : "+str(glo_var.client_data[round_count_in % glo_var.list_len][x][name]))


# 判斷 現在是否有其他條 thread 正在跑後台
# 且會確認遊戲已經寫入後台 可以找到這一局的資料
# finish_time : 繼續遊戲跳出來的時間作為起始
# sleep_time : 經過幾秒後確定後台會有這一筆資料
def can_get_server_data(finish_time, sleep_time = 35) : 
    global glo_var
    
    # 確定回傳回去的時候 已經離遊戲結束 至少sleep_time秒
    delta_time = (datetime.datetime.now() - finish_time).seconds #用當前時間減掉上一局遊戲結束時間
    print("後台 已等待" +str(delta_time)+"秒")
    if delta_time < sleep_time : 
        time.sleep(sleep_time-delta_time) #讓程式確定有足夠sleep_time

    #可以開始執行之前確定 有沒有人在用後台 如果有在用就等一下
    if glo_var.server_using : 
        # 改成sleep
        total_wait_time = 30
        print("上一場的後台還沒跑完 後台等待" + str(total_wait_time) + "秒")
        for x in range(total_wait_time) :
            if x % 10 == 1 :
                print("後台等待剩餘時間 : " + str(total_wait_time-x))
            time.sleep(1)
            if glo_var.server_using == False :
                print("上一場的後台跑完了 開始爬取後台")
                break
    
    # 已經休息過一次了，應該要可以用了，如果還不行，大概就沒救了 
    if glo_var.server_using :
        print("已經休息過一次了，應該要可以用了，如果還不行，大概就沒救了....")
        glo_var.fail_playing = True
        return False

    # 確認以上條件都通過 因此return True
    return True


# 計算等待時間
# limit 超過幾秒算 time out
# state 如果 time out 印出錯誤資訊 要附上是哪個階段錯誤 (是string)
# True == time out
def cal_time_out(limit, state = "") :
    global glo_var
    # 距離紀錄時間多遠
    now_time = datetime.datetime.now()
    delta_time = (now_time - glo_var.record_time).seconds
    # print(delta_time)
    if delta_time >= limit :
        print_to_output(str(state) + " has past " + str(delta_time)+ " sec")
        return True
    else :    
        return False
    
# 單純用來關閉背景音樂
def Game_envi_close_music() :
    pass
    # 如果不想關閉音樂 此區塊註解
    # --------------------------------------------------------------------------------------------------
    # for x in range(5) :
    #     if compare_sim("setting",sys._getframe().f_code.co_name, precise = True, lobby=True) > 0.97 : 
    #         # 通常 break 就會跳出此迴圈 進入下一個 state
    #         click_mid("設定", dosleep = 1)
    #         click((832, 789), "關閉BGM")
    #         click((1339, 795), "關閉音效")
    #         click((1576, 342), "關閉設定")
    #         return 
    # --------------------------------------------------------------------------------------------------

def print_exception(exceptio):
    error_class = exceptio.__class__.__name__ #取得錯誤類型
    detail = exceptio.args[0] #取得詳細內容
    cl, exc, tb = sys.exc_info() #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)
    