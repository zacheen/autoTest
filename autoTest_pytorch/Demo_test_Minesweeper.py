import unittest
import os
from threading import Thread, Event
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

from Minesweeper.Minesweeper_manager import Minesweeper_manager
from RL_Agent import get_agent

class Minesweeper_Begin_thread (Thread):
    def __init__(self) :
        Thread.__init__(self)

    def run(self) :    
        pass
        # Tool_Main.cut_pic_data("player_money_bef", Tool_Main.glo_var.player_num, Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len, cover=False)

# 初始化 遊戲結束要執行的 Thread
class Minesweeper_End_thread (Thread):
    def __init__(self):
        Thread.__init__(self)

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
            if Tool_Main.cal_time_out(10,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break
            
            if Tool_Main.compare_sim("level_beginner",sys._getframe().f_code.co_name) > 0.97 :
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

    def test_click_middle(self):
        Tool_Main.glo_var.s_record_time()

        while True :
            if Tool_Main.cal_time_out(200,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break
            
            if Tool_Main.compare_sim("init_grid",sys._getframe().f_code.co_name) > 0.97 :
            # if Tool_Main.compare_sim("grab_none",sys._getframe().f_code.co_name) > 0.97 :
                Minesweeper_Begin_thread().start()
                break

    def decide_next_step_and_play(self, game_status):
        Tool_Main.glo_var.s_record_time()
        # looping until find a position that is in the game_region
        while True :
            # 1. 截取當前畫面
            game_status.save_pic_path = Tool_Main.cut_pic_data(
                "whole_screen", 
                Tool_Main.glo_var.player_num, 
                0, 
                cover=True, 
                comp=True
            )
        
            # 2. 載入截圖並預處理
            screenshot_path = game_status.save_pic_path[-1]
            current_state = game_status.agent.preprocess_screen(screenshot_path)
        
            # 3. 選擇動作 (輸出 [0,1] 範圍的 x, y)
            action = game_status.agent.select_action(current_state, add_noise=True)
            game_status.update_state(current_state, action)
        
            # 4. 轉換為螢幕座標並點擊
            click_x, click_y = game_status.agent.action_to_screen_coords(action)
            print(f"Step {game_status.step_count}: action=({action[0]:.3f}, {action[1]:.3f}) -> click=({click_x}, {click_y})")
            
            if Tool_Main.click((click_x, click_y), limit_region=game_status.game_region) :
                break

            # if click position is out of game_region 
            # really negitive reward and keep looping
            print("Model decided to click in invalid position")
            game_status.reward = -3.0
            self.update_model(game_status)

    def update_model(self, game_status):
        # 6. 儲存經驗
        if game_status.previous_state is not None and game_status.previous_action is not None:
            game_status.agent.store_transition(
                game_status.previous_state,
                game_status.previous_action,
                game_status.current_state if not game_status.game_over else None,
                game_status.reward,
                game_status.game_over
            )
        
        # 7. 訓練
        loss_info = game_status.agent.train_step()
        if loss_info:
            print(f"  Loss - Critic: {loss_info['critic_loss']:.4f}", end="")
            if loss_info['actor_loss']:
                print(f", Actor: {loss_info['actor_loss']:.4f}")
            else:
                print()

    class Game_status():
        def __init__(self):
            # regions (left, top, width, height)
            screen_region = (0, 0, 1920, 1080) # the size of the screen
            # region limitation [(st_x,st_1,len_n,len_y), have to be inside or outside]
            self.game_region = [((1, 31, 1919, 987), True), ((713, 32, 498, 45), False)]
            
            # 取得 Agent
            self.agent = get_agent(screen_region)
            self.agent.reset_episode()

            self.previous_state = None
            self.previous_action = None
            self.current_state = None
            self.action = None

            # Since might due to unexpected reason, we are not able to keep playing the game
            # EX: cover by other window, the game crush or close ...
            self.max_steps = 30
            self.step_count = 0 # can I use the step in agent??

            self.game_over = False
            self.reward = 0.0

        def update_state(self, new_state, new_action):
            self.previous_state = self.current_state
            self.previous_action = self.action

            self.current_state = new_state
            self.action = new_action

    def test_RL(self):
        Tool_Main.glo_var.s_record_time()
        UI_waiting_time = 1
        game_status = Game_test_case.Game_status()
        time.sleep(UI_waiting_time)
        self.decide_next_step_and_play(game_status)
        time.sleep(UI_waiting_time)
        
        while True:
            time.sleep(1)
            if Tool_Main.glo_var.fail_playing :
                self.assertTrue(False, "time_out")
                break

            last_pic_pos = f"whole_screen_comp_{0+11}_{0}"
            # since a small change in the whole screen shot is tiny, the threshold should be very strick
            if Tool_Main.compare_sim(last_pic_pos,sys._getframe().f_code.co_name, precise = True) < 0.99999 : 
                # case : something changed
                # game status for valid click
                game_status.step_count += 1
                game_status.reward = 20.0
                print("有效點擊！")
                time.sleep(UI_waiting_time)

                # 檢查輸了
                if Tool_Main.compare_sim("lose", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = -5.0
                    game_status.game_over = True
                    print("💥 踩到地雷！")
                
                # 檢查贏了
                elif Tool_Main.compare_sim("win", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = 50.0
                    game_status.game_over = True
                    print("🎉 獲勝！")

                self.update_model(game_status)
                if not game_status.game_over :
                    self.decide_next_step_and_play(game_status)

            elif Tool_Main.cal_time_out(5,sys._getframe().f_code.co_name):
                # check still in game
                if Tool_Main.compare_sim("buttons",sys._getframe().f_code.co_name, precise = True) < 0.99 :
                    # not sure what happens, so don't give reward to model
                    game_status.game_over = True
                    Tool_Main.glo_var.fail_playing = True
                
                # case : nothing change after a period
                game_status.step_count += 1
                game_status.reward = -1.0
                print("無效點擊（畫面無變化）")
                self.update_model(game_status)
                self.decide_next_step_and_play(game_status)
                
                if game_status.step_count > game_status.max_steps:
                    Tool_Main.glo_var.fail_playing = True
            
            # game_over
            if game_status.game_over:
                print(f"Episode 結束: {game_status.agent.get_stats()}")
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

            if Tool_Main.compare_sim("confirm",sys._getframe().f_code.co_name, precise = False) >= 0.9 : 
                # KPSZNN_End_thread().start()
                # CQ9 沒有後台 所以不用等
                # total_wait_time = 100
                # print("等待資料寫入資料庫且辨識完("+ str(total_wait_time)+"秒)")
                # for x in range(total_wait_time) :
                #     if x % 10 == 1 :
                #         print("等待剩餘時間 : " + str(total_wait_time-x))
                #     time.sleep(1)
                # KPSZNN_End_thread().start() # I need to lock here (after screen shot then I can click)
                Tool_Main.click_mid("關閉確認")
                Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len] = str(datetime.datetime.now().strftime(Tool_Main.format_for_db_time))
                Tool_Main.print_to_output("此局結束時間 : " + Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len])
                break

    def test_new_game(self):
        Tool_Main.glo_var.s_record_time()
        while True :
            # 這裡會設 200 是因為我可能會需要切頁面做什麼事情 這個時候可以做
            if Tool_Main.cal_time_out(60,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim("new_game",sys._getframe().f_code.co_name, precise = False) >= 0.97 : 
                Tool_Main.click_mid("新遊戲")
                break
    # 進入遊戲之後 用例增加區↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


Game_envi = "Minesweeper_local_py"
Tool_Main.Game_envi = Game_envi

game_name = "Minesweeper"
player_num = 1
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
    if Game_envi == "Minesweeper_local_py" :
        game_only_var.mine = Minesweeper_manager()
        game_only_var.mine.thread_start()
        print("open the game successfully")
    else :
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
        
        while Tool_Main.glo_var.fail_playing == False:
            # 此區塊是遊戲內的 testcase
            
            # 要先加一 (但有錯的是上一回合 因此兩行下面 report_error 的 round_count 要減一)
            # (有進入遊戲才需要加一) (所以只有這裡才需要加一)
            round_count = round_count+1
            # print("進while迴圈")
            # 初始化測試容器
            during_gameing=unittest.TestSuite() 
            # 組合要做的步驟
            during_gameing.addTest(Game_test_case("test_state_prepare"))
            during_gameing.addTest(Game_test_case("test_click_middle"))
            during_gameing.addTest(Game_test_case("test_RL"))
            during_gameing.addTest(Game_test_case("test_wait_result"))
            during_gameing.addTest(Game_test_case("test_new_game"))
            #獲取當前時間，這樣便於下面的使用
            Tool_Main.glo_var.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

            #打開一個文件，將result寫入此file中 
            fp=open("./testreport/Report-"+Tool_Main.glo_var.file_create_time+"(playing_game)(第"+str(round_count)+"回).html",'wb') 
            runner=HTMLTestRun.HTMLTestRunner(stream=fp,title='KPSZNN',description=u'玩遊戲的測試結果:', file_create_time = Tool_Main.glo_var.file_create_time)
            # 開始執行測是用例
            runner.run(during_gameing)
            fp.close()
        
        sleep_time = 20
        if Tool_Main.glo_var.fail_playing :
            Tool_Main.report_error(round_count)
            game_only_var.mine.thread_stop()
            Tool_Main.print_to_output("fail_playing 等待 "+str(sleep_time)+" 秒")
            time.sleep(sleep_time)
            Tool_Main.print_to_output("重新啟動")
            game_only_var.mine.thread_start()
            Tool_Main.glo_var.reset_var(round_count+1)
            continue
            