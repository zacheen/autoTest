import unittest
import os
from threading import Thread, Event
import datetime
import random
import sys
import time
import traceback
import pyautogui
from pynput import keyboard

IS_PAUSED = False
def on_press(key):
    global IS_PAUSED
    if key == keyboard.Key.end:
        IS_PAUSED = not IS_PAUSED
        if IS_PAUSED:
            print("\n[PAUSED] Press 'End' to resume...")
        else:
            print("\n[RESUMED]")

def check_pause():
    global IS_PAUSED
    while IS_PAUSED:
        time.sleep(0.1)

import HTMLTestRun
import Tool_Main
from Gf_Except import Game_fail_Exception
from Minesweeper_web_client import MinesweeperWebClient

from Minesweeper.Minesweeper_manager import Minesweeper_manager
from visual_discrete_agent import get_agent

WEB_API = MinesweeperWebClient(default_difficulty="Training 6x6")
REWARD_VALID_CLICK = 1.0
REWARD_INVALID_CLICK = -0.98
REWARD_LOSE = -1.0
REWARD_WIN = 3.6

class Minesweeper_Begin_thread (Thread):
    def __init__(self) :
        Thread.__init__(self)

    def run(self) :    
        pass

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
            
            if Tool_Main.compare_sim("level_training",sys._getframe().f_code.co_name) > 0.97 :
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
            
            if Tool_Main.compare_sim("new_game",sys._getframe().f_code.co_name) > 0.97 :
            # if Tool_Main.compare_sim("grab_none",sys._getframe().f_code.co_name) > 0.97 :
                Minesweeper_Begin_thread().start()
                break

    def decide_next_step_and_play(self, game_status):
        Tool_Main.glo_var.s_record_time()
        # looping until find a position that is in the game_region
        while True :
            check_pause()
            # 1. 截取當前畫面
            game_status.save_pic_path = Tool_Main.cut_pic_data(
                "grid_region", 
                Tool_Main.glo_var.player_num, 
                0, 
                cover=True, 
                comp=True
            )
        
            # 2. 載入截圖並預處理
            screenshot_path = game_status.save_pic_path[-1]
            current_screenshot = game_status.agent.preprocess_screen(screenshot_path)
        
            # 3. 選擇動作 (輸出 [0,1] 範圍的 x, y)
            action, log_info = game_status.agent.select_action(
                current_screenshot, 
                add_noise=game_status.noise
            )
            game_status.update_state(current_screenshot, action)
            game_status.log_info = log_info
        
            # 4. 將 36-class action 透過 API 打到網頁版遊戲
            row, col = game_status.agent.action_to_grid(action)
            print(f"Step {game_status.step_count}: action={action} -> ({row},{col}) -> ", end="")
            game_status.click_attempt_count += 1

            if WEB_API.click_cell(row, col):
                game_status.agent.log_action_image(
                    current_screenshot, 
                    log_info, 
                    game_status.step_count
                )
                break

            print("API action failed")
            game_status.reward = -1.0
            game_status.invalid_click_count += 1
            game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)
            
            game_status.agent.log_action_image(
                current_screenshot, 
                log_info, 
                game_status.step_count,
                reward=game_status.reward
            )
            
            self.update_model(game_status)

    def update_model(self, game_status):
        # 儲存經驗
        if game_status.previous_pic is not None and game_status.previous_action is not None:
            game_status.agent.store_transition(
                game_status.previous_pic,
                game_status.previous_action,
                game_status.current_pic if not game_status.game_over else None,
                game_status.reward,
                game_status.game_over
            )

        # 訓練
        loss_info = game_status.agent.train_step()
        if loss_info:
            if 'critic_loss' in loss_info and 'actor_loss' in loss_info:
                print(f"Loss - Critic: {loss_info['critic_loss']:.4f}, Actor: {loss_info['actor_loss']:.4f}")
            else:
                print(f"Loss: {loss_info['loss']:.4f}, Q Mean: {loss_info['q_mean']:.4f}")

    class Game_status():
        def __init__(self):
            # regions (left, top, width, height)
            grid_region = (745, 361, 432, 434) # the size of the screen
            # region limitation [(st_x,st_1,len_n,len_y), have to be inside or outside]
            self.game_region = [((1, 31, 1919, 987), True), ((713, 32, 498, 45), False)]
            
            # 取得 Agent
            self.agent = get_agent(grid_region)
            self.agent.reset_episode()

            self.previous_pic = None
            self.previous_action = None
            self.current_pic = None
            self.action = None
            self.log_info = None

            # Since might due to unexpected reason, we are not able to keep playing the game
            # EX: cover by other window, the game crush or close ...
            self.max_steps = 60
            self.step_count = 0 # can I use the step in agent??

            self.game_over = 0 # Since this will pass into the model, and 0 represents not game over, 1 represents game over
            self.reward = 0.0
            self.invalid_click_count = 0
            self.click_attempt_count = 0
            self.won = False

        def update_state(self, new_state, new_action):
            self.previous_pic = self.current_pic
            self.previous_action = self.action

            self.current_pic = new_state
            self.action = new_action

        def invalid_click_rate(self):
            if self.click_attempt_count <= 0:
                return 0.0
            return self.invalid_click_count / self.click_attempt_count

    def test_RL(self):
        Tool_Main.glo_var.s_record_time()
        UI_waiting_time = 1
        game_status = Game_test_case.Game_status()
        game_status.noise = True  # SAC handles exploration via stochastic policy
        time.sleep(UI_waiting_time)
        self.decide_next_step_and_play(game_status)
        time.sleep(UI_waiting_time)

        while True:
            check_pause()
            time.sleep(1)
            if game_status.game_over :
                game_status.agent.log_episode_metrics(
                    win=game_status.won,
                    invalid_click_rate=game_status.invalid_click_rate(),
                )
                game_status.agent.on_episode_end()
                self.assertTrue(True, "game_over(really finish the game)")
                break
            elif Tool_Main.glo_var.fail_playing :
                game_status.agent.log_episode_metrics(
                    win=False,
                    invalid_click_rate=game_status.invalid_click_rate(),
                )
                game_status.agent.on_episode_end()
                self.assertTrue(False, "time_out(reach max steps)")
                break

            last_pic_pos = f"grid_region_comp_{0+11}_{0}"
            # since a small change in the whole screen shot is tiny, the threshold should be very strick
            if Tool_Main.compare_sim(last_pic_pos,sys._getframe().f_code.co_name, precise = True) < 0.9995 :
                # case : something changed
                # game status for valid click
                game_status.step_count += 1
                game_status.reward = REWARD_VALID_CLICK
                print("有效點擊！")
                time.sleep(UI_waiting_time)

                # 檢查輸了
                if Tool_Main.compare_sim("lose", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = REWARD_LOSE
                    game_status.game_over = 1
                    print("踩到地雷！")

                # 檢查贏了
                elif Tool_Main.compare_sim("win", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = REWARD_WIN
                    game_status.game_over = 1
                    game_status.won = True
                    print("獲勝！")

                game_status.agent.clear_blocked_actions(reason="screen changed after valid click")
                self.update_model(game_status)
                if not game_status.game_over :
                    self.decide_next_step_and_play(game_status)

            elif Tool_Main.cal_time_out(2,sys._getframe().f_code.co_name):
                # check still in game
                if Tool_Main.compare_sim("buttons",sys._getframe().f_code.co_name, precise = True) < 0.99 :
                    # not sure what happens, so don't give reward to model
                    game_status.game_over = 1

                # case : nothing change after a period
                game_status.step_count += 1
                game_status.reward = REWARD_INVALID_CLICK
                game_status.invalid_click_count += 1
                print("無效點擊（畫面無變化）")
                if game_status.current_pic is not None and game_status.action is not None:
                    game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)
                self.update_model(game_status)
                if game_status.step_count > game_status.max_steps:
                    Tool_Main.glo_var.fail_playing = True
                else :
                    self.decide_next_step_and_play(game_status)

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
                # Tool_Main.click_mid("關閉確認") # website version don't have confirm button
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
                if WEB_API.start_new_game():
                    break
    # 進入遊戲之後 用例增加區↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


Game_envi = "Minesweeper_web"
Tool_Main.Game_envi = Game_envi

game_name = "Minesweeper_web"
player_num = 1
# 初始化全部遊戲都會用到的參數

next_wait_time = 60

if __name__=="__main__" : 
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

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
        print("now in Minesweeper_local_py successfully")
    elif Game_envi == "Minesweeper_web" :
        Tool_Main.open_game_web()
        print("now in Minesweeper_web successfully")
    else :
        raise Exception(f"Game_envi {Game_envi} doesn't exist!")


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
        
        sleep_time = 3
        if Tool_Main.glo_var.fail_playing :
            Tool_Main.report_error(round_count)
            if Game_envi == "Minesweeper_local_py" :
                game_only_var.mine.thread_stop()
            elif Game_envi == "Minesweeper_web" :
                Tool_Main.glo_var.game_driver.quit()
            Tool_Main.print_to_output("fail_playing 等待 "+str(sleep_time)+" 秒")
            time.sleep(sleep_time)
            Tool_Main.print_to_output("重新啟動")
            if Game_envi == "Minesweeper_local_py" :
                game_only_var.mine.thread_start()
            elif Game_envi == "Minesweeper_web" :
                Tool_Main.open_game_web()
            Tool_Main.glo_var.reset_var(round_count+1)
            continue
            
