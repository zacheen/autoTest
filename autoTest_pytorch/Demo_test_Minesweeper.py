import unittest
import os
from threading import Thread, Event
import datetime
import random
import sys
import time
import traceback

from util.Log import Logger, print_to_output, report_error, format_for_db_time

# AUTOTEST_NO_KEYBOARD=1 skips pynput on X-less environments (e.g., Colab) where
# Listener crashes at startup. Without a listener, IS_PAUSED stays False and
# check_pause() is a no-op, so the demo loses only the End-key pause shortcut.
NO_KEYBOARD_LISTENER = bool(os.environ.get("AUTOTEST_NO_KEYBOARD"))
if not NO_KEYBOARD_LISTENER:
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

import util.HTMLTestRun as HTMLTestRun
import util.Tool_Main as Tool_Main
from util.Gf_Except import Game_fail_Exception
from Minesweeper_web_client import MinesweeperWebClient

from Minesweeper.Minesweeper_manager import Minesweeper_manager
from model_structure.eval_utils import (
    finish_eval_timing,
    should_run_eval,
    start_eval_timing,
)
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from visual_discrete_agent_v3 import get_agent


EVAL_INTERVAL = 200
EVAL_EPISODES = 30
EVAL_OFFSET = 50
EVAL_MAX_STEPS_PER_EPISODE = 200
EVAL_STEP_WAIT_SECONDS = 0.1


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

        global glo_var
        glo_var.state.round_count_for_pipe += 1
        pass_in_round_count_for_pipe = glo_var.state.round_count_for_pipe
        print(f"Starting round {pass_in_round_count_for_pipe}")

        Tool_Main.cut_pic_data(glo_var, "player_money_aft", glo_var.state.player_num, glo_var.state.slot(glo_var.state.round_count),          cover=False)
        Tool_Main.cut_pic_data(glo_var, "win_lose",         glo_var.state.player_num, glo_var.state.slot(pass_in_round_count_for_pipe), cover=False)
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
        #     Logger.pipe_write("Pipeline round " + str(pass_in_round_count_for_pipe) + " — backend crawl error\n")

        #     raise Game_fail_Exception

        # Tool_Main.glo_var.server_using = False
        KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe)

def KPSZNN_do_compare(server_data, pass_in_round_count_for_pipe):
    global game_only_var
    global glo_var
    slot  = glo_var.state.slot(pass_in_round_count_for_pipe)
    print_to_output("Round " + str(pass_in_round_count_for_pipe))
    print_to_output("KPSZNN_do_compare client data: " + str(glo_var.state.client_data[slot]))
    Logger.pipe_write("Pipeline round " + str(pass_in_round_count_for_pipe) + "\n")

# Globals used only by this main script.
class Game_only_var() :
    def __init__(self) :
        pass

class Game_test_case(unittest.TestCase) :
    _last_eval_started_at = None

    @classmethod
    def setUpClass(self):
        # Per-game init values.
        # Values here reset every round.
        pass

    def test_choose_room(self):
        global glo_var
        glo_var.state.set_record_time()

        while True :
            if Tool_Main.cal_time_out(glo_var, 10, sys._getframe().f_code.co_name) or glo_var.state.fail_playing :
                glo_var.state.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim(glo_var, "level_training", sys._getframe().f_code.co_name, threshold=0.97) :
            # if Tool_Main.compare_sim(glo_var, "roomLV1", sys._getframe().f_code.co_name, threshold=0.97) :
                Tool_Main.click_mid(glo_var, "click room")
                break

    # ── test cases - after entering game ──────────────────────────

    def test_state_prepare(self) :
        global glo_var
        glo_var.state.set_record_time()
        glo_var.state.round_count += 1
        slot = glo_var.state.slot(glo_var.state.round_count)
        glo_var.state.begin_time[slot] = str((datetime.datetime.now()+datetime.timedelta(minutes=-3)).strftime(format_for_db_time))
        print_to_output(f"Round {glo_var.state.round_count}, start at {glo_var.state.begin_time[slot]}")

        # take screenshot for html report
        Tool_Main.compare_sim(glo_var, "", sys._getframe().f_code.co_name)

    def test_click_middle(self):
        global glo_var
        glo_var.state.set_record_time()

        while True :
            if Tool_Main.cal_time_out(glo_var, 200, sys._getframe().f_code.co_name) or glo_var.state.fail_playing :
                glo_var.state.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim(glo_var, "new_game", sys._getframe().f_code.co_name, threshold=0.97) :
            # if Tool_Main.compare_sim(glo_var, "grab_none", sys._getframe().f_code.co_name, threshold=0.97) :
                Minesweeper_Begin_thread().start()
                break

    def decide_next_step_and_play(self, game_status):
        Tool_Main.glo_var.state.set_record_time()  # glo_var.state.set_record_time() is a thin delegate to state
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
            game_status.reward = MINESWEEPER_REWARD_CONFIG.invalid_click
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
        global glo_var
        game_status.save_pic_path = Tool_Main.cut_pic_data(
            glo_var,
            "grid_region",
            glo_var.state.player_num,
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

    def maybe_run_eval(self, agent):
        episode = agent.episode_count
        if not should_run_eval(episode, offset=EVAL_OFFSET, interval=EVAL_INTERVAL):
            return

        eval_started_at, seconds_since_last_eval = start_eval_timing(
            Game_test_case._last_eval_started_at
        )
        Game_test_case._last_eval_started_at = eval_started_at

        print(
            f"[V3 EVAL] Start fixed-policy evaluation at episode {episode}: "
            f"{EVAL_EPISODES} episodes"
        )
        eval_stats = self.run_fixed_policy_evaluation(agent, EVAL_EPISODES)
        duration_seconds = finish_eval_timing(eval_started_at)

        agent.log_eval_metrics(
            avg_reward=eval_stats["avg_reward"],
            win_rate=eval_stats["win_rate"],
            avg_steps=eval_stats["avg_steps"],
            avg_invalid_rate=eval_stats["avg_invalid_rate"],
            seconds_since_last_eval=seconds_since_last_eval,
            duration_seconds=duration_seconds,
        )

    def run_fixed_policy_evaluation(self, agent, num_episodes):
        rewards = []
        wins = []
        steps = []
        invalid_rates = []

        for eval_idx in range(1, num_episodes + 1):
            stats = self.run_eval_episode(agent)
            rewards.append(stats["reward"])
            wins.append(1 if stats["is_win"] else 0)
            steps.append(stats["steps"])
            invalid_rates.append(stats["invalid_rate"])
            print(
                f"[V3 EVAL] {eval_idx:>2}/{num_episodes}: "
                f"{'WIN ' if stats['is_win'] else 'LOSE'} | "
                f"reward={stats['reward']:.3f} | "
                f"steps={stats['steps']} | "
                f"invalid={stats['invalid_rate']:.2%}"
            )

        return {
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "win_rate": sum(wins) / max(len(wins), 1) * 100.0,
            "avg_steps": sum(steps) / max(len(steps), 1),
            "avg_invalid_rate": sum(invalid_rates) / max(len(invalid_rates), 1),
        }

    def run_eval_episode(self, agent):
        if not WEB_API.start_new_game():
            raise RuntimeError("[V3 EVAL] Failed to start eval game")

        game_status = Game_test_case.Game_status()
        game_status.agent = agent
        game_status.noise = False
        game_status.server_state = WEB_API.get_game_state()
        agent.clear_blocked_actions(reason="eval episode reset")

        done = False
        while not done and game_status.step_count < EVAL_MAX_STEPS_PER_EPISODE:
            check_pause()
            current_screenshot = self.capture_grid_state(game_status)
            action, _ = agent.select_action(current_screenshot, add_noise=False)
            game_status.update_state(current_screenshot, action)
            row, col = agent.action_to_grid(action)
            game_status.click_attempt_count += 1

            api_result = WEB_API.click_cell_with_state(row, col)
            if not api_result or not api_result.get("ok"):
                game_status.reward = MINESWEEPER_REWARD_CONFIG.invalid_click
                game_status.invalid_click_count += 1
                game_status.record_reward(game_status.reward)
                agent.block_action_for_state(current_screenshot, action)
                game_status.step_count += 1
                time.sleep(EVAL_STEP_WAIT_SECONDS)
                continue

            next_server_state = api_result.get("data")
            board_changed = (
                self._board_signature(next_server_state) !=
                self._board_signature(game_status.server_state)
            )
            game_status.server_state = next_server_state

            server_status = game_status.server_state.get("status")
            if server_status == "lost":
                game_status.reward = MINESWEEPER_REWARD_CONFIG.lose
                game_status.game_over = 1
                done = True
            elif server_status == "won":
                game_status.reward = MINESWEEPER_REWARD_CONFIG.win
                game_status.game_over = 1
                game_status.won = True
                done = True
            elif board_changed:
                game_status.reward = MINESWEEPER_REWARD_CONFIG.valid_click
            else:
                game_status.reward = MINESWEEPER_REWARD_CONFIG.invalid_click
                game_status.invalid_click_count += 1
                agent.block_action_for_state(current_screenshot, action)

            game_status.record_reward(game_status.reward)
            game_status.step_count += 1
            time.sleep(EVAL_STEP_WAIT_SECONDS)

        return {
            "reward": game_status.total_reward,
            "steps": game_status.step_count,
            "is_win": game_status.won,
            "invalid_rate": game_status.invalid_click_rate(),
        }

    def test_RL(self):
        global glo_var
        glo_var.state.set_record_time()
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
                self.maybe_run_eval(game_status.agent)
                self.assertTrue(True, "game_over(really finish the game)")
                break
            elif glo_var.state.fail_playing :
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
            if Tool_Main.compare_sim(glo_var, last_pic_pos, sys._getframe().f_code.co_name, precise=True, threshold=0.97, disappear=True) :
                # case : something changed
                # game status for valid click
                game_status.step_count += 1
                game_status.reward = MINESWEEPER_REWARD_CONFIG.valid_click
                print("valid click")
                time.sleep(UI_waiting_time)

                # check loss
                if Tool_Main.compare_sim(glo_var, "lose", sys._getframe().f_code.co_name, precise=True, threshold=0.9):
                    game_status.reward = MINESWEEPER_REWARD_CONFIG.lose
                    game_status.game_over = 1
                    print("hit mine")

                # check win
                elif Tool_Main.compare_sim(glo_var, "win", sys._getframe().f_code.co_name, precise=True, threshold=0.9):
                    game_status.reward = MINESWEEPER_REWARD_CONFIG.win
                    game_status.game_over = 1
                    game_status.won = True
                    print("win")

                game_status.record_reward(game_status.reward)
                game_status.next_state = self.capture_grid_state(game_status)
                self.update_model(game_status)
                if not game_status.game_over :
                    self.decide_next_step_and_play(game_status)

            elif Tool_Main.cal_time_out(glo_var, 2, sys._getframe().f_code.co_name):
                # check still in game
                if Tool_Main.compare_sim(glo_var, "buttons", sys._getframe().f_code.co_name, precise=True, threshold=0.97, disappear=True) :
                    # not sure what happens, so don't give reward to model
                    game_status.game_over = 1

                # case : nothing change after a period
                game_status.step_count += 1
                game_status.reward = MINESWEEPER_REWARD_CONFIG.invalid_click
                game_status.record_reward(game_status.reward)
                game_status.invalid_click_count += 1
                print("invalid click (no screen change)")
                if game_status.current_pic is not None and game_status.action is not None:
                    game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)
                game_status.next_state = game_status.current_pic
                self.update_model(game_status)
                if game_status.step_count > game_status.max_steps:
                    glo_var.state.fail_playing = True
                else :
                    self.decide_next_step_and_play(game_status)

    def test_RL_server(self):
        """RL training loop driven by server state instead of screenshot diffing.

        Flow per step:
          1. capture screenshot → YOLO/policy → choose action
          2. POST /click via WEB_API → receive next server_state
          3. classify reward based on server_state["status"] + board diff
          4. capture next screenshot for replay buffer
          5. agent.maybe_train_step()
        """
        global glo_var
        glo_var.state.set_record_time()
        UI_waiting_time = 1
        game_status = Game_test_case.Game_status()
        game_status.noise = True
        game_status.server_state = WEB_API.get_game_state()
        time.sleep(UI_waiting_time)
        self.decide_next_step_and_play(game_status)
        time.sleep(UI_waiting_time)

        while True:
            check_pause()
            time.sleep(0.1)
            if game_status.game_over:
                game_status.agent.log_episode_metrics(
                    win=game_status.won,
                    invalid_click_rate=game_status.invalid_click_rate(),
                    reward_mean=game_status.average_reward(),
                )
                game_status.agent.on_episode_end()
                self.maybe_run_eval(game_status.agent)
                self.assertTrue(True, "game_over(really finish the game)")
                break
            elif glo_var.state.fail_playing:
                game_status.agent.log_episode_metrics(
                    win=False,
                    invalid_click_rate=game_status.invalid_click_rate(),
                    reward_mean=game_status.average_reward(),
                )
                game_status.agent.on_episode_end()
                self.assertTrue(False, "time_out(reach max steps)")
                break

            if game_status.pending_server_state is None:
                continue

            game_status.step_count += 1
            game_status.server_state = game_status.pending_server_state
            game_status.pending_server_state = None
            board_changed = game_status.pending_board_changed
            game_status.pending_board_changed = False

            server_status = game_status.server_state.get("status")
            if server_status == "lost":
                game_status.reward = MINESWEEPER_REWARD_CONFIG.lose
                game_status.game_over = 1
                print("lose")
            elif server_status == "won":
                game_status.reward = MINESWEEPER_REWARD_CONFIG.win
                game_status.game_over = 1
                game_status.won = True
                print("win")
            elif board_changed:
                game_status.reward = MINESWEEPER_REWARD_CONFIG.valid_click
                print("valid click")
            else:
                game_status.reward = MINESWEEPER_REWARD_CONFIG.invalid_click
                game_status.invalid_click_count += 1
                print("invalid click")

            game_status.record_reward(game_status.reward)

            if board_changed:
                if not game_status.game_over:
                    game_status.next_state = self.capture_grid_state(game_status)
                else:
                    game_status.next_state = None
            else:
                if game_status.current_pic is not None and game_status.action is not None:
                    game_status.agent.block_action_for_state(game_status.current_pic, game_status.action)
                game_status.next_state = game_status.current_pic

            self.update_model(game_status)
            if game_status.step_count > game_status.max_steps:
                glo_var.state.fail_playing = True
            elif not game_status.game_over:
                self.decide_next_step_and_play(game_status)

    def test_wait_result(self):
        global glo_var
        glo_var.state.set_record_time()
        while True :
            if Tool_Main.cal_time_out(glo_var, 3, sys._getframe().f_code.co_name) or glo_var.state.fail_playing :
                glo_var.state.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim(glo_var, "confirm", sys._getframe().f_code.co_name, precise=False, threshold=0.9) :
                # < call the process when the game is ended, usually include backend data crawling and checking data correctness >
                # KPSZNN_End_thread().start()
                # total_wait_time = 100
                # print(f"waiting for backend process to finish for {str(total_wait_time)} seconds")
                # for x in range(total_wait_time) :
                #     if x % 10 == 1 :
                #         print(f"remain waiting time : {str(total_wait_time-x)} seconds")
                #     time.sleep(1)
                # Tool_Main.click_mid(glo_var, "click confirm button") # website version don't have confirm button
                slot = glo_var.state.slot(glo_var.state.round_count)
                glo_var.state.end_time[slot] = str(datetime.datetime.now().strftime(format_for_db_time))
                print_to_output("Round end time: " + glo_var.state.end_time[slot])
                break

    def test_new_game(self):
        global glo_var
        glo_var.state.set_record_time()
        while True :
            if Tool_Main.cal_time_out(glo_var, 3, sys._getframe().f_code.co_name) or glo_var.state.fail_playing :
                glo_var.state.fail_playing = True
                self.assertTrue(False,"time_out")
                break

            if Tool_Main.compare_sim(glo_var, "new_game", sys._getframe().f_code.co_name, precise=False, threshold=0.97) :
                if WEB_API.start_new_game():
                    break
    # ── end of test cases - after entering game ──────────────────────────

GAME_ENV = "Minesweeper_web"
GAME_NAME = "Minesweeper_web"
player_num = 1

if __name__=="__main__" :
    # Start keyboard listener (skipped on X-less envs via AUTOTEST_NO_KEYBOARD)
    if not NO_KEYBOARD_LISTENER:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    round_count = 1
    from util.Info.Glo_var import Glo_var
    glo_var = Glo_var(
        game_name = GAME_NAME,
        game_env = GAME_ENV,
        player_num = player_num,
        round_count = round_count
    )
    Tool_Main.set_glo_var(glo_var)

    game_only_var = Game_only_var()
    round_count = round_count-1
    if GAME_ENV == "Minesweeper_local_py" :
        game_only_var.mine = Minesweeper_manager()
        game_only_var.mine.thread_start()
        print("now in Minesweeper_local_py successfully")
    elif GAME_ENV == "Minesweeper_web" :
        glo_var.driver = Tool_Main.open_game_web()
        WEB_API = MinesweeperWebClient(glo_var.driver, default_difficulty="Training 6x6")
        print("now in Minesweeper_web successfully")
    else :
        raise Exception(f"GAME_ENV {GAME_ENV} doesn't exist!")


    # main loop — runs forever, restart on error
    while True :
        # outer loop: pre-game test cases (lobby / room selection)
        open_game=unittest.TestSuite()
        open_game.addTest(Game_test_case("test_choose_room"))

        glo_var.state.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        fp=open(f"./testreport/Report-{glo_var.state.file_create_time}(open_game).html",'wb')
        runner=HTMLTestRun.HTMLTestRunner(stream=fp,title=GAME_NAME,description=u'Report for opening game:', file_create_time = glo_var.state.file_create_time)
        # start to run the test case
        runner.run(open_game)
        fp.close()

        while glo_var.state.fail_playing == False:
            # test cases in the game
            round_count = round_count+1
            during_gameing=unittest.TestSuite()
            # combine the test cases (usually is the game flow)
            during_gameing.addTest(Game_test_case("test_state_prepare"))
            during_gameing.addTest(Game_test_case("test_click_middle"))
            # during_gameing.addTest(Game_test_case("test_RL"))
            during_gameing.addTest(Game_test_case("test_RL_server"))
            during_gameing.addTest(Game_test_case("test_wait_result"))
            during_gameing.addTest(Game_test_case("test_new_game"))

            glo_var.state.file_create_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            fp=open(f"./testreport/Report-{glo_var.state.file_create_time}(playing_game) ({round_count} round).html",'wb')
            runner=HTMLTestRun.HTMLTestRunner(stream=fp,title=GAME_NAME,description=u'Report for playing game:', file_create_time = glo_var.state.file_create_time)
            # start to run the test cases
            runner.run(during_gameing)
            fp.close()

        sleep_time = 3
        if glo_var.state.fail_playing :
            report_error(round_count)
            if GAME_ENV == "Minesweeper_local_py" :
                game_only_var.mine.thread_stop()
            elif GAME_ENV == "Minesweeper_web" :
                glo_var.driver.quit()
            print_to_output(f"fail_playing detected. Waiting {sleep_time}s...")
            time.sleep(sleep_time)
            print_to_output("Restarting.")
            if GAME_ENV == "Minesweeper_local_py" :
                game_only_var.mine.thread_start()
            elif GAME_ENV == "Minesweeper_web" :
                glo_var.driver = Tool_Main.open_game_web()
                WEB_API = MinesweeperWebClient(glo_var.driver, default_difficulty="Training 6x6")
            glo_var.reset(round_count+1)
            continue
            
