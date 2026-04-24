from util.Click import ClickPyautogui

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import time
import pyautogui

class Chrome_Driver:
    def __init__(self, game_env):
        """Open browser, log in, and wire the driver into the session."""
        print("open browser")
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1960,1080")
        options.add_argument("disable-infobars")
        prefs = {
            "": "",
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
        }
        options.add_experimental_option("prefs", prefs)

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.game_env = game_env
        ClickPyautogui.click(30, 30)
        time.sleep(1)
        self.full_screen()
        self.login_plat()

    def login_plat(self):
        print("login platform : ", self.game_env)
        if self.game_env == "CQ9":
            self.driver.get("https://h5bt.cqgame.games/h5/BT02/?language=zh-cn&?token=guest")
        elif self.game_env == "Minesweeper_web":
            self.driver.get("http://127.0.0.1:8000")
        else:
            raise Exception(f"Game_env {self.game_env} doesn't exist!")
    
    def full_screen(self):
        self.driver.maximize_window()
        pyautogui.hotkey("f11")

    def open_book_mark():
        pyautogui.hotkey("ctrl", "shift", "b")

    def switch_to_game_web(self):
        """Call this after the page opens a new tab, so selenium can control it."""
        print("switching web page")
        self.driver.switch_to.window(self.driver.window_handles[-1])

        