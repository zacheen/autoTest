import sys
import traceback
import datetime
import time
from bs4 import BeautifulSoup
import pandas as pd
from os import system

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options 
from selenium.common.exceptions import WebDriverException 
import pickle

import os
import compare_server

#自己創建的工具
import Card
from Gf_Except import Game_fail_Exception
import catch_back as catch_back #後台的Tool_Main


def search_D21D(fp, b_date, e_date, ID_in = None, in_num = 1, write_new_playing_num = False):
    if catch_back.driver == None :
        catch_back.prepare_server(catch_back.envir, catch_back.account, catch_back.password)

    error_count = 0

    # 讀取已經跑過的局號
    already_run_number = []
    if os.path.isfile("./back_log/already_run.txt") : 
        with open('./back_log/already_run.txt', 'r') as f :
            while True :
                read_in = f.readline()
                if read_in == "" :
                    break
                else:
                    already_run_number.append(read_in.strip())

    while True :
        
        fail_exception = False
        
        # print("get_screenshot_as_file")
        # catch_back.driver.get_screenshot_as_file('./ttest.png')
        
        if fail_exception == False : 
            try :           
                # print("進 while 現在 error_count = "+str(error_count))
                catch_back.switch_to_win_lose()

            except WebDriverException as e : 
                # 只要driver == None  就代表我尚未初始化 driver
                catch_back.driver = None
                # 錯誤記數+1
                error_count = error_count+1
                # 報錯
                catch_back.print_error(e, error_count, "輸贏報表無法按下去")
                # 像Main一樣 是為了跳過下面的步驟
                fail_exception = True

        if fail_exception == False :
            try :
                catch_back.fill_win_lose_search(b_date = b_date, e_date = e_date, ID_in = ID_in, game_name = None)
                try:
                    result_len = catch_back.get_result_len()
                except IndexError :
                    print("此頁面沒有資料")
                    result_len = 0

            except WebDriverException as e : 
                error_count = error_count+1
                catch_back.print_error(e, error_count, "找不到輸入位置")
                fail_exception = True

        server_data_list = []
        if write_new_playing_num :
            already_run_f = open('./back_log/already_run.txt', 'a')
        other_id_f = open('./back_log/other_id.txt', 'a')
        total_hour = {}
        total_hour["count"] = 0
        total_hour["valid_beg"] = 0
        total_hour["win_lose"] = 0
        for index in range(1,min(in_num, result_len)+1) : 
            # print("index :", index)
            index_mod_20 = index%20
            if index_mod_20 == 0 :
                index_mod_20 = 20

            if index >= 21 and index % 20 == 1:
                catch_back.next_page()
                # print("換頁成功")   

            if fail_exception == False :
                try:
                    playing_number = catch_back.get_playing_number(index_mod_20)
                    # print(playing_number)

                    if playing_number in already_run_number :
                        print(playing_number,"此局號已經測試過了")
                        with open('./back_log/overlap.txt', 'a') as f_overlap :
                            f_overlap.write(str(playing_number)+"此局號已經測試過了!!\n")
                        continue
                    else :
                        # print("繼續")

                        # 確認玩家id規則
                        player_id = catch_back.get_player_id(index_mod_20)

                        try :
                            daili = player_id.split("_")
                            daili = int(daili[0])
                        except :
                            other_id_f.write(player_id +"不屬於任何代理 因此跳過\n")
                            daili = 0 
                        # id_pass = False
                        # for cha in player_id :
                        #     if cha > "0" and cha < "9" :
                        #         continue
                        #     elif cha == "_" :
                        #         id_pass = True
                        #         break
                        #     else :
                        #         break
                        
                        # print("繼續2")
                        # if id_pass == False:
                        #     print(player_id +"不屬於任何代理 因此跳過")
                        #     continue 
                    
                        # server_data 裡面就只有 期间有效投注, 期间盈利, 局号
                        server_data = {}
                        already_run_number.append(playing_number)
                        if write_new_playing_num :
                            already_run_f.write(playing_number+"\n")
                        server_data['局号'] = playing_number
                        # print("繼續3")  
                        server_data['有效投注'] = compare_server.float_to_int_100(catch_back.get_valid_beg(index_mod_20),100000)
                        server_data['盈利'] = compare_server.float_to_int_100(catch_back.get_win_lose(index_mod_20),100000)
                        # print("繼續4")

                        #統計
                        if daili not in total.keys() :
                            total[daili] = {}
                            total[daili]["count"] = 0
                            total[daili]["valid_beg"] = 0
                            total[daili]["win_lose"] = 0

                        total[daili]["count"] += 1
                        total[daili]["valid_beg"] += server_data['有效投注']
                        total[daili]["win_lose"] += server_data['盈利']

                        total_hour["count"] += 1
                        total_hour["valid_beg"] += server_data['有效投注']
                        total_hour["win_lose"] += server_data['盈利']

                except WebDriverException as e : 
                    error_count = error_count+1
                    catch_back.print_error(e, error_count, "抓取對局日誌失敗")
                    fail_exception = True

            if fail_exception == False :
                try:
                    # catch_back.switch_to_same_table(index_mod_20)
                    # 這裡還沒抓取 same_table 的資料
                    # 這裡一定要記得關 因為此頁面的名稱與輸贏頁面一部分重複了
                    # get_same_table_data(server_data, fp)
                    # catch_back.close_same_table()
                    # catch_back.back_to_win_lose()

                    # print("後台資料擷取成功")
                    #最後記得要把資料存入list中
                    server_data_list.append(server_data)
                    # 測試cal money用 把結果存起來 測試D21D_cal_money的時候可以直接讀這個 驗證有沒有錯
                    catch_back.write_pickle("對局日誌的server_data",server_data)
                    pass

                except WebDriverException as e : 
                    error_count = error_count+1
                    catch_back.print_error(e, error_count, "抓取同桌失敗")
                    fail_exception = True

        if write_new_playing_num :
            already_run_f.flush()
        other_id_f.flush()

        print(total_hour)
        # input()

        if fail_exception == False :
            try:
                # 以下兩行是為了 還原到初始狀態 為了下一次可以順利進行
                catch_back.driver.switch_to.default_content()
                error_count = 0

                return server_data_list , in_num >= result_len

            except WebDriverException as e : 
                error_count = error_count+1
                catch_back.print_error(e, error_count, "還原到初始狀態失敗")
                fail_exception = True

        if error_count >= 2 :
            error_count = 0
            catch_back.quit_server()
            print("catch_back error")
            fail_exception = False
            raise Game_fail_Exception

        if error_count >= 1 :
            # time.sleep(500)
            fail_exception = False
            catch_back.prepare_server(catch_back.envir, catch_back.account, catch_back.password)



if __name__ == "__main__" :
    import sys
    # catch_back.TW = "nc2"
    catch_back.TW = 1
    # sys.stdout 代表cmd 輸出
    # ---------------------------------------------------
    # 單獨測試拆解字串function
    # print(D21D_get_log_data(1, sys.stdout))
    
    #-----------------------------------------------------
    
    # 最大執行數量 如果大於搜尋數量 就會以搜尋數量為主
    total_num = 3
    
    # 每抓幾筆資料印出來一次 (如果某一筆資料出錯其他筆資料並不會印出來)
    one_round = 20
    
    pipe_output_f = open("./user_change/pipe_output.txt","w")

    which_day = "2021-06-07 "

    catch_back.prepare_server("XX","XX","XX")
    print("finish prepare server")

    #for x in range(1,(total_num//one_round)+2):

    total = {}
    
    for x in range(24):
        if x <= 9 :
            server_data, do_break = search_D21D(sys.stdout, b_date=which_day+"0"+str(x)+":00", e_date=which_day+"0"+str(x)+":59", in_num=999999, write_new_playing_num = True)
        else:
            server_data, do_break = search_D21D(sys.stdout, b_date=which_day    +str(x)+":00", e_date=which_day    +str(x)+":59", in_num=999999, write_new_playing_num = True)
        # 把已經跑過的局號紀錄下來
        

        # # 統計
        # for i in range(len(server_data)) : 
        #     total_count += 1
        #     total_valid_beg += server_data[i]['有效投注']
        #     total_win_lose += server_data[i]['盈利']
        pipe_output_f.flush()
        # print("finish")

    print(total)
    # print("total_count", total_count)
    # print("total_valid_beg", total_valid_beg)
    # print("total_win_lose", total_win_lose)
        # if do_break :
        #     break
    
    
    
    
   