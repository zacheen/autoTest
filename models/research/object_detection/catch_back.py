import sys
import traceback
import datetime
import time
from bs4 import BeautifulSoup
import pandas as pd
from os import system
import os
import pickle

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from Card import Card 
from Gf_Except import Game_fail_Exception

Game_envi = 1

now_path = os.getcwd()
user_abs_loc = now_path + "\\user_change\\"
webdriver_path = user_abs_loc + "chromedriver.exe"
# print(webdriver_path)

catch_back_error_f = open(user_abs_loc+'catch_back_error.txt', "a", encoding='UTF-8')
account = "XX"
password = "XX"
envir = "XX"
driver = None

wait_time = 1
when_net_suck = 1


def print_back_state(str_in):
    # print(str_in)
    pass

def print_error(e, error_count, why = ""):
    global catch_back_error_f
    
    error_class = e.__class__.__name__ #取得錯誤類型
    detail = e.args[0] #取得詳細內容
    cl, exc, tb = sys.exc_info() #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)
    
    ISOTIMEFORMAT = '%Y_%m_%d %H:%M:%S'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    print("website error !!!!!!!!!!!!!!!!!!!!!!!!")
    
    catch_back_error_f.write("第"+str(error_count)+"次 website error !!!!!!!!!!!!!!!!!!!!!!! at "+theTime+"\n" )

    if why != "" :
        print("也許是因為 : " + why)
        catch_back_error_f.write("也許是因為 : " + why + "\n")

    catch_back_error_f.flush()

def prepare_server(envir_in = None, account_in = None, password_in = None):
    global envir
    global account
    global password

    if envir_in != None:
        envir = envir_in
    else :
        envir_in = envir
    if account_in != None:
        account = account_in
    else :
        account_in = account
    if password_in != None:
        password = password_in
    else :
        password_in = password

    global driver
    if driver == None : 
        print("後台 prepare_server  driver == None ")
        initial()
        open_website(envir_in)
        login(account_in, password_in)
        open_side_bar()
    else : 
        print("後台 prepare_server  driver != None ")
        open_website(envir_in)
        open_side_bar()

def initial() : 
    global driver
    if driver == None:
        options = Options()
        # options.headless = True
        options.add_argument("--window-size=1160,700")
        options.add_argument('disable-infobars')
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        prefs = {"":""}
        prefs["credentials_enable_service"] = False
        prefs["profile.password_manager_enabled"] = False
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(executable_path=webdriver_path, options=options)
        # driver.set_window_position(1921, 0)
        time.sleep(1)
        driver.maximize_window()
    else : 
        print("error !!!!!!!!!!!!!!!!!! Main hit because driver is using")
        raise Game_fail_Exception

def open_website(envir_in):
    global driver
    global envir
    if Game_envi == "CQ9":
        print("CQ9 demo 沒有後台")
    elif Game_envi == "某個 Game_envi":
        driver.get("http://192.168.X.X:port") # 後台網址
    else :
        print("Game_envi錯誤")

def login(account_in, password_in):
    global account
    global password
    global driver

    account = account_in
    password = password_in

    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#login")))
    print_back_state("找到送出")

    server_ID=driver.find_element_by_name("MERCHANT_USERNAME")
    server_ID.send_keys(account)#輸入帳號
    
    server_PASSWORD=driver.find_element_by_name("MERCHANT_PWD")
    server_PASSWORD.send_keys(password)#請輸入密碼
    time.sleep(0.2)
    
    commit=driver.find_element_by_css_selector("#login")
    commit.click()


def open_side_bar():
    global driver
    if Game_envi == 1  or Game_envi == 2:
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#side-menu > li:nth-child(6) > a')))
        print_back_state("找到報表管理")
        report=driver.find_element_by_css_selector("#side-menu > li:nth-child(6) > a")
        report.click()
        time.sleep(0.5)
    elif Game_envi == "nc2" or Game_envi == "nc1":
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR , '#side-menu > li:nth-child(6) > a')))
        print_back_state("找到報表管理")
        report=driver.find_element_by_css_selector("#side-menu > li:nth-child(6) > a")
        report.click()
        time.sleep(0.5)
    else :
        print("Game_envi錯誤")
# #切到輸贏報表
# def switch_to_win_lose():
#     global driver
#     WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#side-menu > li.active > ul > li:nth-child(2) > a')))
#     print_back_state("找到輸贏報表")
#     LoseWin_report=driver.find_element_by_css_selector("#side-menu > li.active > ul > li:nth-child(2) > a")
#     LoseWin_report.click()

#     WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@src,"/winAndLoseReport")]')))
#     driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport")]'))
#切到輸贏報表
def switch_to_win_lose():
    global driver
    if Game_envi == 1  or Game_envi == 2:
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#side-menu > li.active > ul > li:nth-child(2) > a')))
        print_back_state("找到輸贏報表")
        LoseWin_report=driver.find_element_by_css_selector("#side-menu > li.active > ul > li:nth-child(2) > a")
        
    elif Game_envi == "nc2" or Game_envi == "nc1" :
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#side-menu > li.active > ul > li:nth-child(2) > a')))
        print_back_state("找到輸贏報表")
        LoseWin_report=driver.find_element_by_css_selector("#side-menu > li.active > ul > li:nth-child(2) > a")
        
    LoseWin_report.click()

    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@src,"/winAndLoseReport")]')))
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport")]'))

def fill_win_lose_search(b_date = None, e_date= None, ID_in= None, game_name= None):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#btnSearch')))
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.ID, 'selGameType')))
    
    if b_date != None:
        #輸入開始日期
        driver.find_element_by_name("BEGIN_DATE").send_keys(b_date)
    if e_date != None:
        #輸入結束日期
        driver.find_element_by_name("END_DATE").send_keys(e_date)
    if ID_in != None:
        #輸入會員編號
        driver.find_element_by_name("txtAccounts").send_keys(ID_in)
    if game_name != None:
        game_typ_ele = Select(driver.find_element_by_id("selGameType"))
        time.sleep(1)
        game_typ_ele.select_by_visible_text(game_name)
        # Select(driver.find_element_by_id("selGameType")).select_by_visible_text(game_name)
    time.sleep(0.2)

    driver.find_element_by_css_selector("#btnSearch").click()

def get_result_len():
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#no-more-tables > div.bootstrap-table > div.fixed-table-container > div.fixed-table-pagination > div.pull-left.pagination-detail > span.pagination-info")))
    time.sleep(wait_time+1)
    num_info = driver.find_element_by_css_selector("#no-more-tables > div.bootstrap-table > div.fixed-table-container > div.fixed-table-pagination > div.pull-left.pagination-detail > span.pagination-info").text
    print("num_info : ", num_info.split(" "))
    num_info = num_info.split(" ")[-2]
    # print("num_info ,",num_info)
    return int(num_info)

def get_player_id(index = 1):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(2)")))
    return driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(2)").text
                                                #tblData > tbody > tr:nth-child(1)              > td:nth-child(2)
def get_valid_beg(index = 1):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(10)")))
    return driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(10)").text

def get_win_lose(index = 1):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(11)")))
    return driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(11)").text

def get_playing_number(index = 1):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(14) > a")))
    return driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(14) > a").text

# 抓取對局日志 並轉資料
def get_log_str(index = 1):
    global driver
    switch_to_gameLog(index)
    time.sleep(when_net_suck)
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.ID, "OpValue")))
    time.sleep(when_net_suck)
    # soup=BeautifulSoup(driver.page_source,"lxml")
    # game_log=soup.find("textarea",class_="form-control")
    game_log = driver.find_element_by_xpath('/html/body/div/div/div/div/div/div/div[2]/div[2]/div/div/pre')
    return game_log.text

# def switch_to_gameLog(index = 1):
#     global driver
#     WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)")))
#     driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)").click()
#     driver.switch_to.default_content()
#     WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src,'/gameLog')]")))
#     driver.switch_to.frame(driver.find_element_by_xpath("//iframe[contains(@src,'/gameLog')]"))  # gameLog 是日誌
def switch_to_gameLog(index = 1):
    global driver
    if Game_envi == 1  or Game_envi == 2 :
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)")))
        driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)").click()
    elif Game_envi == "nc2" or Game_envi == "nc1" :
        WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)")))
        driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)").click()
        
    else :
        print("Game_envi錯誤")

    driver.switch_to.default_content()
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src,'/gameLog')]")))
    driver.switch_to.frame(driver.find_element_by_xpath("//iframe[contains(@src,'/gameLog')]"))  # gameLog 是日誌

def back_to_win_lose() :
    global driver
    driver.switch_to.parent_frame()
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#li_81 > a")))
    driver.find_element_by_css_selector("#li_81 > a").click()
    time.sleep(0.5)
    driver.switch_to.default_content()
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@src,"/winAndLoseReport")]')))
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport")]'))  # 切換回輸贏報表

def next_page():
    time.sleep(when_net_suck)
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#no-more-tables > div.bootstrap-table > div.fixed-table-container > div.fixed-table-pagination > div.pull-right.pagination > ul > li.page-next > a")))
    driver.find_element_by_css_selector("#no-more-tables > div.bootstrap-table > div.fixed-table-container > div.fixed-table-pagination > div.pull-right.pagination > ul > li.page-next > a").click()
    driver.switch_to.default_content()
    time.sleep(2)
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@src,"/winAndLoseReport")]')))
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport")]'))  # 切換回輸贏報表
    # /html/body/div[2]/div/div/div[2]/iframe

def switch_to_gameresult(index = 1) :
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(14) > a")))
    driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(14) > a").click()
    driver.switch_to.default_content()
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@src,"/gameResult?gameNo")]')))
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/gameResult?gameNo")]'))   # gameResult 是點局號跳出的遊戲結果

    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.ID, "tblData")))


# def switch_to_same_table(index = 1):
#     global driver
#     WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(1)")))
#     driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(1)").click()
#     driver.switch_to.default_content()
#     time.sleep(0.5)
#     driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport/sameTable?")]'))
def switch_to_same_table(index = 1):
    global driver
    WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(1)")))
    # WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)")))
    driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(1)").click()
    # driver.find_element_by_css_selector("#tblData > tbody > tr:nth-child("+str(index)+") > td:nth-child(15) > a:nth-child(2)").click()
    driver.switch_to.default_content()
    time.sleep(0.5)
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[contains(@src,"/winAndLoseReport/sameTable?")]'))

    # WebDriverWait(driver, 8*wait_time, 0.5).until(EC.presence_of_element_located((By.ID, "tblSameTableData")))


def close_same_table():
    driver.switch_to.default_content()
    driver.find_element_by_css_selector("#li_10002 > a > button").click()
    time.sleep(when_net_suck)

# 把整個object(可以是dict)寫入檔案儲存起來
def write_pickle(name, server_data):
    # 寫 object
    with open(name+".pickle", 'wb') as f:
        pickle.dump(server_data, f)

# 把整個object(可以是dict)讀入 return 
def read_pickle(name, server_data):
    # 讀 object
    with open(name+'.pickle', 'rb') as f:
        read_object = pickle.load(f)
    return read_object

def quit_server():
    global driver
    try :
        driver.quit()
    except AttributeError :
        print("driver 已變成 None")
        pass
    driver = None