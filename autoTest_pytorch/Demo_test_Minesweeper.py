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
from visual_discrete_agent_v2 import get_agent

WEB_API = MinesweeperWebClient(default_difficulty="Training 6x6")
REWARD_VALID_CLICK = 1.0
REWARD_INVALID_CLICK = -0.5
# design corrosponding to discount factor = 0.7
# -0.5*0.7*0.7 + -0.5*0.7 + -0.5 = -1.095
# a little bit less than REWARD_LOSE, since I hope model learn not to click invalid position
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
        print(f"Starting round {pass_in_round_count_for_pipe}")
        
        Tool_Main.cut_pic_data("player_money_aft", Tool_Main.glo_var.player_num, Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len, cover=False)
        Tool_Main.cut_pic_data("win_lose"        , Tool_Main.glo_var.player_num, pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len, cover=False)
        print("finish screen shot card type")

        print("start identifing")
        # Tool_Main.set_client_data("player_money_bef", "Initial score", 10 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len )
        # Tool_Main.set_client_data("player_money_aft", "Finial score", 10 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len )
        # Tool_Main.set_client_data("win_lose"        , "Win/Loss Score", 12 , pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len , thresh = 0.3)

        print("Finish identifing. Start initiating backend data crawling") # CQ9, Minesweeper don't have backend data
        server_data = None
        # if Tool_Main.can_get_server_data(finish_time) : 
        #     Tool_Main.glo_var.server_using = True
        # else :
        #     print("Error when crawling backend data in KPSZNN_End_thread")
        #     raise Game_fail_Exception
 
        # try : 
        #     server_data = KPSZNN_catch_back.search_KPSZNN(b_date=Tool_Main.glo_var.begin_time[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len], e_date=Tool_Main.glo_var.end_time[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len],ID_in = str(Tool_Main.glo_var.game_agent_ID)+"_"+str(Tool_Main.glo_var.game_account))
        # except Game_fail_Exception as e : 
        #     print("Error when crawling backend data in KPSZNN_End_thread")
        #     error_class = e.__class__.__name__
        #     detail = e.args[0]
        #     cl, exc, tb = sys.exc_info()
        #     lastCallStack = traceback.extract_tb(tb)[-1]
        #     fileName = lastCallStack[0]
        #     lineNum = lastCallStack[1]
        #     funcName = lastCallStack[2]
        #     errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        #     print(errMsg)

        #     Tool_Main.glo_var.fail_playing = True
        #     Tool_Main.glo_var.server_using = True
        #     Tool_Main.glo_var.pipe_output_f.write("Pipeline round " + str(pass_in_round_count_for_pipe) + " — backend crawl error\n")

        #     raise Game_fail_Exception

        # Tool_Main.glo_var.server_using = False
        KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe)

def KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe):
    global game_only_var
    Tool_Main.print_to_output("Round " + str(pass_in_round_count_for_pipe))
    Tool_Main.print_to_output("KPSZNN_do_compare client data: "+str(Tool_Main.glo_var.client_data[pass_in_round_count_for_pipe%Tool_Main.glo_var.list_len]))
    Tool_Main.glo_var.pipe_output_f.write("Pipeline round " + str(pass_in_round_count_for_pipe) + "\n")

        # 這裡放的是 只有這個 Main 會用到的全域變數
class Game_only_var() :
    def __init__(self) :
        pass

class Game_test_case(unittest.TestCase) :
    @classmethod
    def setUpClass(self):
        # 這邊放每一場都要 初始化 的 參數
        # 這裡的值每回合遊戲都會重置一次(資料會不見)
        # 開始時執行
        pass

    def test_choose_room(self):
        Tool_Main.glo_var.set_record_time()

        while True :
            if Tool_Main.cal_time_out(10,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break
            
            if Tool_Main.compare_sim("level_training",sys._getframe().f_code.co_name) > 0.97 :
            # if Tool_Main.compare_sim("roomLV1",sys._getframe().f_code.co_name) > 0.97 :
                Tool_Main.click_mid("click room")
                break

    # ── test cases - after entering game ──────────────────────────

    def test_state_prepare(self) :
        Tool_Main.glo_var.set_record_time()
        Tool_Main.glo_var.round_count += 1
        Tool_Main.glo_var.begin_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len] = str((datetime.datetime.now()+datetime.timedelta(minutes=-3)).strftime(Tool_Main.format_for_db_time))
        Tool_Main.print_to_output(f"Round {str(Tool_Main.glo_var.round_count)}, start at {Tool_Main.glo_var.begin_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len]}")

        # take screenshot for html report
        Tool_Main.compare_sim("", sys._getframe().f_code.co_name)

    def test_click_middle(self):
        Tool_Main.glo_var.set_record_time()

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
        Tool_Main.glo_var.set_record_time()
        # looping until find a position that is in the game_region
        while True :
            check_pause()
            current_screenshot = self.capture_grid_state(game_status)
            action, log_info = game_status.agent.select_action(
                current_screenshot,
                add_noise=game_status.noise
            )
            game_status.update_state(current_screenshot, action)
            game_status.log_info = log_info

            row, col = game_status.agent.action_to_grid(action)
            print(f"Step {game_status.step_count}: action={action} -> ({row},{col}) -> ", end="")
            game_status.click_attempt_count += 1

            api_result = WEB_API.click_cell_with_state(row, col)
            if api_result and api_result.get("ok"):
                next_server_state = api_result.get("data")
                game_status.pending_server_state = next_server_state
                game_status.pending_board_changed = (
                    self._board_signature(next_server_state) !=
                    self._board_signature(game_status.server_state)
                )
                game_status.agent.log_action_image(
                    current_screenshot,
                    log_info,
                    game_status.step_count
                )
                break

            print("API action failed")
            game_status.reward = REWARD_INVALID_CLICK
            game_status.record_reward(game_status.reward)
            game_status.invalid_click_count += 1
            game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)

            game_status.agent.log_action_image(
                current_screenshot,
                log_info,
                game_status.step_count,
                reward=game_status.reward
            )

            game_status.next_state = current_screenshot
            self.update_model(game_status)

    def capture_grid_state(self, game_status):
        game_status.save_pic_path = Tool_Main.cut_pic_data(
            "grid_region",
            Tool_Main.glo_var.player_num,
            0,
            cover=True,
            comp=True
        )
        screenshot_path = game_status.save_pic_path[-1]
        return game_status.agent.preprocess_screen(screenshot_path)

    def _board_signature(self, server_state):
        if not server_state:
            return None
        board = server_state.get("board")
        if board is None:
            return None
        return tuple(
            tuple((cell.get("state"), cell.get("value")) for cell in row)
            for row in board
        )

    def update_model(self, game_status):
        state = game_status.current_pic
        action = game_status.action
        next_state = game_status.next_state
        reward = game_status.reward
        done = bool(game_status.game_over)
        game_status.agent.store_transition(
            state,
            action,
            next_state if not done else None,
            reward,
            done,
        )

        loss_info = game_status.agent.maybe_train_step(force=done)
        if loss_info:
            if 'critic_loss' in loss_info and 'actor_loss' in loss_info:
                print(f"Loss - Critic: {loss_info['critic_loss']:.4f}, Actor: {loss_info['actor_loss']:.4f}")
            elif 'Q_loss' in loss_info and 'q_mean' in loss_info:
                print(f"Q Loss: {loss_info['Q_loss']:.4f}, Q Mean: {loss_info['q_mean']:.4f}")
            else:
                print(f"Train metrics: {loss_info}")

    class Game_status():
        def __init__(self):
            # regions (left, top, width, height)
            grid_region = (745, 361, 432, 434) # the size of the screen
            # region limitation [(st_x,st_1,len_n,len_y), have to be inside or outside]
            self.game_region = [((1, 31, 1919, 987), True), ((713, 32, 498, 45), False)]
            
            self.agent = get_agent(grid_region)
            self.agent.reset_episode()


            self.current_pic = None
            self.action = None
            self.next_state = None
            self.log_info = None

            # Since might due to unexpected reason, we are not able to keep playing the game
            # EX: cover by other window, the game crush or close ...
            self.max_steps = 60
            self.step_count = 0 # can I use the step in agent??

            self.game_over = 0 # Since this will pass into the model, and 0 represents not game over, 1 represents game over
            self.reward = 0.0
            self.total_reward = 0.0
            self.reward_count = 0
            self.invalid_click_count = 0
            self.click_attempt_count = 0
            self.won = False
            self.server_state = None
            self.pending_server_state = None
            self.pending_board_changed = False

        def update_state(self, new_state, new_action):
            self.current_pic = new_state
            self.action = new_action

        def invalid_click_rate(self):
            if self.click_attempt_count <= 0:
                return 0.0
            return self.invalid_click_count / self.click_attempt_count

        def record_reward(self, reward):
            self.total_reward += float(reward)
            self.reward_count += 1

        def average_reward(self):
            if self.reward_count <= 0:
                return 0.0
            return self.total_reward / self.reward_count

    def test_RL(self):
        Tool_Main.glo_var.set_record_time()
        UI_waiting_time = 1
        game_status = Game_test_case.Game_status()
        game_status.noise = True  # enable epsilon-greedy during play
        time.sleep(UI_waiting_time)
        self.decide_next_step_and_play(game_status)

        while True:
            check_pause()
            if game_status.game_over :
                game_status.agent.log_episode_metrics(
                    win=game_status.won,
                    invalid_click_rate=game_status.invalid_click_rate(),
                    reward_mean=game_status.average_reward(),
                )
                game_status.agent.on_episode_end()
                self.assertTrue(True, "game_over(really finish the game)")
                break
            elif Tool_Main.glo_var.fail_playing :
                game_status.agent.log_episode_metrics(
                    win=False,
                    invalid_click_rate=game_status.invalid_click_rate(),
                    reward_mean=game_status.average_reward(),
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
                print("valid click")
                time.sleep(UI_waiting_time)

                # check loss
                if Tool_Main.compare_sim("lose", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = REWARD_LOSE
                    game_status.game_over = 1
                    print("hit mine")

                # check win
                elif Tool_Main.compare_sim("win", sys._getframe().f_code.co_name, precise=True) >= 0.9:
                    game_status.reward = REWARD_WIN
                    game_status.game_over = 1
                    game_status.won = True
                    print("win")

                game_status.record_reward(game_status.reward)
                game_status.agent.clear_blocked_actions(reason="screen changed after valid click")
                game_status.next_state = self.capture_grid_state(game_status)
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
                game_status.record_reward(game_status.reward)
                game_status.invalid_click_count += 1
                print("invalid click (no screen change)")
                if game_status.current_pic is not None and game_status.action is not None:
                    game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)
                game_status.next_state = game_status.current_pic
                self.update_model(game_status)
                if game_status.step_count > game_status.max_steps:
                    Tool_Main.glo_var.fail_playing = True
                else :
                    self.decide_next_step_and_play(game_status)

    def test_wait_result(self):
        Tool_Main.glo_var.set_record_time()
        while True :
            if Tool_Main.cal_time_out(3,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim("confirm",sys._getframe().f_code.co_name, precise = False) >= 0.9 : 
                # < call the process when the game is ended, usually include backend data crawling and checking data correctness >
                # KPSZNN_End_thread().start()
                # total_wait_time = 100
                # print(f"waiting for backend process to finish for {str(total_wait_time)} seconds")
                # for x in range(total_wait_time) :
                #     if x % 10 == 1 :
                #         print(f"remain waiting time : {str(total_wait_time-x)} seconds")
                #     time.sleep(1)
                # Tool_Main.click_mid("click confirm button") # website version don't have confirm button
                Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len] = str(datetime.datetime.now().strftime(Tool_Main.format_for_db_time))
                Tool_Main.print_to_output("此局結束時間 : " + Tool_Main.glo_var.end_time[Tool_Main.glo_var.round_count%Tool_Main.glo_var.list_len])
                break

    def test_new_game(self):
        Tool_Main.glo_var.set_record_time()
        while True :
            if Tool_Main.cal_time_out(3,sys._getframe().f_code.co_name) or Tool_Main.glo_var.fail_playing :
                Tool_Main.glo_var.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim("new_game",sys._getframe().f_code.co_name, precise = False) >= 0.97 : 
                if WEB_API.start_new_game():
                    break
    # ── end of test cases - after entering game ──────────────────────────

Game_envi = "Minesweeper_web"
Tool_Main.Game_envi = Game_envi

# initialize all the parameter relevant to all games
game_name = "Minesweeper_web"
player_num = 1

if __name__=="__main__" : 
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    round_count = 1
    Tool_Main.glo_var = Tool_Main.Glo_var(
        in_game_name = game_name, 
        player_num = player_num,           # player_num is related to the number of screen shots
        round_count = round_count
    )
    print("initialize the parameter only for this game")
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


    # main loop — runs forever, restart on error
    while True :
        # outer loop: pre-game test cases (lobby / room selection)
        open_game=unittest.TestSuite()
        open_game.addTest(Game_test_case("test_choose_room"))

        Tool_Main.glo_var.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        fp=open(f"./testreport/Report-{Tool_Main.glo_var.file_create_time}(open_game).html",'wb')
        runner=HTMLTestRun.HTMLTestRunner(stream=fp,title=game_name,description=u'Report for opening game:', file_create_time = Tool_Main.glo_var.file_create_time)
        # start to run the test case
        runner.run(open_game)
        fp.close()
        
        while Tool_Main.glo_var.fail_playing == False:
            # test cases in the game
            round_count = round_count+1
            during_gameing=unittest.TestSuite() 
            # combine the test cases (usually is the game flow)
            during_gameing.addTest(Game_test_case("test_state_prepare"))
            during_gameing.addTest(Game_test_case("test_click_middle"))
            during_gameing.addTest(Game_test_case("test_RL"))
            during_gameing.addTest(Game_test_case("test_wait_result"))
            during_gameing.addTest(Game_test_case("test_new_game"))

            Tool_Main.glo_var.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            fp=open(f"./testreport/Report-{Tool_Main.glo_var.file_create_time}(playing_game) ({str(round_count)} round).html",'wb')
            runner=HTMLTestRun.HTMLTestRunner(stream=fp,title=game_name,description=u'Report for playing game:', file_create_time = Tool_Main.glo_var.file_create_time)
            # start to run the test cases
            runner.run(during_gameing)
            fp.close()
        
        sleep_time = 3
        if Tool_Main.glo_var.fail_playing :
            Tool_Main.report_error(round_count)
            if Game_envi == "Minesweeper_local_py" :
                game_only_var.mine.thread_stop()
            elif Game_envi == "Minesweeper_web" :
                Tool_Main.glo_var.game_driver.quit()
            Tool_Main.print_to_output(f"Errfail_playing detected. Waiting {sleep_time}s...")
            time.sleep(sleep_time)
            Tool_Main.print_to_output("Restarting.")
            if Game_envi == "Minesweeper_local_py" :
                game_only_var.mine.thread_start()
            elif Game_envi == "Minesweeper_web" :
                Tool_Main.open_game_web()
            Tool_Main.glo_var.reset_var(round_count+1)
            continue
            
