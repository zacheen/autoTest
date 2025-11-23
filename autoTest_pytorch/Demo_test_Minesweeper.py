import unittest
import os
import threading
import datetime
import random
import sys
import time
import traceback
import pyautogui

import HTMLTestRun
import Tool_Main
# from Card import Card

from Gf_Except import Game_fail_Exception

class Minesweeper_Begin_thread (threading.Thread):
    def __init__(self) :
        threading.Thread.__init__(self)

    def run(self) :    
        Tool_Main.cut_pic_data("player_money_bef", Tool_Main.glo_var.player_num, Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len, cover=False)

# 初始化 遊戲結束要執行的 Thread
class Minesweeper_End_thread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        finish_time = datetime.datetime.now()
                
        Tool_Main.glo_var.round_count_for_pipe += 1
        pass_in_round_count_for_pipe = Tool_Main.glo_var.round_count_for_pipe
        print("開始執行第"+str(pass_in_round_count_for_pipe)+"回 背景執行 比較後台")
        
        Tool_Main.cut_pic_data("player_money_aft", Tool_Main.glo_var.player_num, Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len, cover=False) #擷取玩家一開始的分數
        Tool_Main.cut_pic_data("win_lose"        , Tool_Main.glo_var.player_num, pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len, cover=False)
        print("截 牌型圖片結束")

        print("截圖完成 開始辨識")
        # Tool_Main.set_client_data("player_money_bef", "携带分數", 10 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len )
        # Tool_Main.set_client_data("player_money_aft", "結束分數", 10 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len )
        # Tool_Main.set_client_data("win_lose"        , "输赢分數", 12 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len , thresh = 0.3)

        print("辨識完成 開始爬後台") # CQ9 沒有後台資料
        server_data = None
        # # 看後台有沒有人在用
        # if Tool_Main.can_get_server_data(finish_time) : 
        #     Tool_Main.glo_var.server_using = True
        # else :
        #     print("爬後台等待時出問題 in KPSZNN_End_thread")
        #     raise Game_fail_Exception
 
        # try : 
        #     server_data = KPSZNN_catch_back.search_KPSZNN(b_date=Tool_Main.glo_var.begin_time[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len], e_date=Tool_Main.glo_var.end_time[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len],ID_in = str(Tool_Main.glo_var.game_agent_ID)+"_"+str(Tool_Main.glo_var.game_account))
        # except Game_fail_Exception as e : 
        #     print("爬後台時出問題 in KPSZNN_End_thread")
        #     error_class = e.__class__.__name__ #取得錯誤類型
        #     detail = e.args[0] #取得詳細內容
        #     cl, exc, tb = sys.exc_info() #取得Call Stack
        #     lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
        #     fileName = lastCallStack[0] #取得發生的檔案名稱
        #     lineNum = lastCallStack[1] #取得發生的行號
        #     funcName = lastCallStack[2] #取得發生的函數名稱
        #     errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        #     print(errMsg)

        #     Tool_Main.glo_var.fail_playing = True
        #     Tool_Main.glo_var.server_using = True
        #     Tool_Main.glo_var.pipe_output_f.write("這是pipeline的第" + str(pass_in_round_count_for_pipe) + "回合  爬取後台錯誤!!!\n" )

        #     raise Game_fail_Exception

        # Tool_Main.glo_var.server_using = False
        KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe)

def KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe):
    global game_only_var
    Tool_Main.print_to_output("第"+str(pass_in_round_count_for_pipe)+"回合")
    # Tool_Main.print_to_output("KPSZNN_do_compare 收到資料(後台) : "+str(server_data))
    Tool_Main.print_to_output("KPSZNN_do_compare 收到資料(前端) : "+str(Tool_Main.glo_var.client_data[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len]))
    Tool_Main.glo_var.pipe_output_f.write("這是pipeline的第" + str(pass_in_round_count_for_pipe) + "回合\n" )

    error_result = ""
    warning_result = ""

class Game_only_var() : 
    def __init__(self) : 
        # 這裡放的是 只有這個 Main 會用到的全域變數
        pass

# 初始化 要執行的動作
class Game_test_case(unittest.TestCase) :
    @classmethod
    def setUpClass(self):
        # 這邊放每一場都要 初始化 的 參數
        # 這裡的值每回合遊戲都會重置一次(資料會不見)
        # 開始時執行
        pass

    def test_choose_room(self):
        Tool_Main.glo_var.s_record_time()

        while True :
            if Tool_Main.cal_time_out(200,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break
            
            if Tool_Main.compare_sim("roomLV6",sys._getframe().f_code.co_name) > 0.97 :
            # if Tool_Main.compare_sim("roomLV1",sys._getframe().f_code.co_name) > 0.97 :
                Tool_Main.click_mid("點擊房間")
                break

    # 進入遊戲之後 用例增加區↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    def test_state_prepare(self) : 
        Tool_Main.glo_var.s_record_time()
        Tool_Main.glo_var.round_count += 1 #確保每一回一開始，回合數可以加1(讓他確定是從1開始)，但如果沒有在初始化的時候先-1，可能會出現預設值為1時，直接加1會直接變成2當作第一回合
        # minutes=-1 是時間會減一分鐘(確保搜尋時間包含這一局的起始時間)
        Tool_Main.glo_var.begin_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len] = str((datetime.datetime.now()+datetime.timedelta(minutes=-3)).strftime(Tool_Main.format_for_db_time))
        Tool_Main.print_to_output("在主程式的第 " + str(Tool_Main.glo_var.round_count) + " 回合")
        Tool_Main.print_to_output("此局開始時間 : " + Tool_Main.glo_var.begin_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len])

        # 單純截圖指令(會放在 html report中)
        Tool_Main.compare_sim("", sys._getframe().f_code.co_name)

    def test_grab_none(self):
        Tool_Main.glo_var.s_record_time()

        while True :
            if Tool_Main.cal_time_out(200,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break
            
            if Tool_Main.compare_sim("grab3",sys._getframe().f_code.co_name) > 0.97 :
            # if Tool_Main.compare_sim("grab_none",sys._getframe().f_code.co_name) > 0.97 :
                KPSZNN_Begin_thread().start()
                time.sleep(2)
                Tool_Main.click_mid("點擊不搶庄")
                break

    # 等待遊戲結束
    def test_wait_result(self):
        Tool_Main.glo_var.s_record_time()

        see_continue = False
        while True :
            # 這裡會設 200 是因為我可能會需要切頁面做什麼事情 這個時候可以做
            if Tool_Main.cal_time_out(60,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim("continue",sys._getframe().f_code.co_name, precise = True) > 0.97 : 
                KPSZNN_End_thread().start()
                # CQ9 沒有後台 所以不用等
                # total_wait_time = 100
                # print("等待資料寫入資料庫且辨識完("+ str(total_wait_time)+"秒)")
                # for x in range(total_wait_time) :
                #     if x % 10 == 1 :
                #         print("等待剩餘時間 : " + str(total_wait_time-x))
                #     time.sleep(1)

                Tool_Main.click_mid("繼續遊戲")
                Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len] = str(datetime.datetime.now().strftime(Tool_Main.format_for_db_time))
                Tool_Main.print_to_output("此局結束時間 : " + Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len])
                break

    # 進入遊戲之後 用例增加區↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


Game_envi = "My_Minesweeper"
Tool_Main.Game_envi = Game_envi

game_name = "Minesweeper"
player_num = 4
# 初始化全部遊戲都會用到的參數

next_wait_time = 60

if __name__=="__main__" : 
    print("完成import全部東西 開始執行 Main")
    # round_count 記數 (用來記現在跑到第幾回合)  (只要程式哪裡有問題或跳error 就要reset Glo_var 的 round_count)
    round_count = 1
    Tool_Main.glo_var = Tool_Main.Glo_var(
        in_game_name = game_name, 
        player_num = player_num,           # 玩家數量最大數量 通常是截圖看要截幾張
        round_count = round_count
    )
    print("開始初始化此遊戲必要變數")
    # 初始化這個遊戲才會用到的參數
    game_only_var = Game_only_var()
    round_count = round_count-1
    print("Tool_Main.glo_var : ",Tool_Main.glo_var)
    print("打開遊戲網頁")
    Tool_Main.open_game_web()
    print("登入遊戲平台")
    Tool_Main.login_plat()


    # 這裡是無窮while迴圈 要讓他可以一直執行
    while True :
        # 這一層是進入遊戲之前的 testcase 
        # (因為進入遊戲之後可以按繼續遊戲 沒有必要回到大廳) 
        # (但很多遊戲有問題之後 等待遊戲結束後 按下刷新 會回到大廳頁面 而不是遊戲頁面 因此有問題要 break出來)
        
        # 初始化測試容器
        open_game=unittest.TestSuite() 

        #將測試用例加入到測試容器中
        open_game.addTest(Game_test_case("test_choose_room"))

        #獲取當前時間，這樣便於下面的使用
        # print("print(Tool_Main.glo_var)",Tool_Main.glo_var)
        Tool_Main.glo_var.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

        #打開file，將result寫入此file中 
        fp=open("./testreport/Report-"+Tool_Main.glo_var.file_create_time+"(open_game).html",'wb')
        runner=HTMLTestRun.HTMLTestRunner(stream=fp,title=game_name,description=u'打開遊戲的測試結果:', file_create_time = Tool_Main.glo_var.file_create_time)
        # 開始執行測是用例
        runner.run(open_game)
        fp.close()

        # 這一整塊是如果有發生錯誤 要做什麼事
        # 重整時間(一局結束的時間) 
        sleep_time = 100
        if Tool_Main.glo_var.fail_playing : 
            # 報錯到 txt
            Tool_Main.report_error("開場")
            # 睡眠(為了等待遊戲結束)
            
            Tool_Main.print_to_output("fail_playing 等待 "+str(sleep_time)+" 秒")
            time.sleep(sleep_time)
            # 網頁刷新
            Tool_Main.print_to_output("重新啟動")
            Tool_Main.glo_var.game_driver.refresh() # 按下網頁刷新鍵
            # 參數reset
            Tool_Main.glo_var.reset_var(round_count)
            # game_only_var.fail_reset()
            continue
        # 驗證開到遊戲前有沒有出錯
        
        
        while True:
            # 此區塊是遊戲內的 testcase
            
            # 要先加一 (但有錯的是上一回合 因此兩行下面 report_error 的 round_count 要減一)
            # (有進入遊戲才需要加一) (所以只有這裡才需要加一)
            round_count = round_count+1
            # print("進while迴圈")
            # 初始化測試容器
            during_gameing=unittest.TestSuite() 
            # 組合要做的步驟
            during_gameing.addTest(Game_test_case("test_state_prepare"))
            during_gameing.addTest(Game_test_case("test_grab_none"))
            during_gameing.addTest(Game_test_case("test_wait_result"))
            #獲取當前時間，這樣便於下面的使用
            Tool_Main.glo_var.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

            #打開一個文件，將result寫入此file中 
            fp=open("./testreport/Report-"+Tool_Main.glo_var.file_create_time+"(playing_game)(第"+str(round_count)+"回).html",'wb') 
            runner=HTMLTestRun.HTMLTestRunner(stream=fp,title='KPSZNN',description=u'玩遊戲的測試結果:', file_create_time = Tool_Main.glo_var.file_create_time)
            # 開始執行測是用例
            runner.run(during_gameing)
            fp.close()

            
            if Tool_Main.glo_var.fail_playing :
                Tool_Main.report_error(round_count)
                Tool_Main.print_to_output("fail_playing 等待 "+str(sleep_time)+" 秒")
                time.sleep(sleep_time)
                Tool_Main.print_to_output("重新啟動")
                Tool_Main.glo_var.game_driver.refresh() # 按下網頁刷新鍵
                Tool_Main.glo_var.reset_var(round_count+1)
                # game_only_var.fail_reset()
                break

        
        if Tool_Main.glo_var.fail_playing :
            Tool_Main.report_error(round_count)
            sleep_time = 50
            Tool_Main.print_to_output("fail_playing 等待 "+str(sleep_time)+" 秒")
            time.sleep(sleep_time)
            Tool_Main.print_to_output("重新啟動")
            Tool_Main.glo_var.game_driver.refresh() # 按下網頁刷新鍵
            Tool_Main.glo_var.reset_var(round_count+1)
            break
            