from util.Click import ClickPyautogui

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

import sys
import time
import pyautogui

class Chrome_Driver:
    def __init__(self, game_env):
        """Open browser, log in, and wire the driver into the session."""
        print("open browser")
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1960,1080")
        # Pin DPR=1 so get_screenshot_as_png() pixels match the .txt region coords
        options.add_argument("--force-device-scale-factor=1.25")
        options.add_argument("disable-infobars")
        # Suppress the "Chrome is being controlled by automated test software" infobar.
        # Without this it eats ~70px at the top of the screen, making viewport ≠ screen
        # and breaking the coord pass-through for both clicks and screenshots.
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
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
            url = "https://h5bt.cqgame.games/h5/BT02/?language=zh-cn&?token=guest"
        elif self.game_env == "Minesweeper_web":
            url = "http://127.0.0.1:8000"
        else:
            raise Exception(f"Game_env {self.game_env} doesn't exist!")

        try:
            self.driver.get(url)
        except WebDriverException as e:
            # Connection failed — most common cause is the local server isn't running yet
            msg = str(e)
            print("=" * 60)
            if "ERR_CONNECTION_REFUSED" in msg:
                print(f"[ERROR] Cannot connect to {url} (connection refused)")
                print("Reason: no server is listening on this port.")
                if self.game_env == "Minesweeper_web":
                    print("Fix: start the Minesweeper server in another terminal first:")
                    print("  conda activate py310_torch251_cuda118")
                    print("  python autoTest_pytorch/Minesweeper_web/server.py")
            elif "ERR_NAME_NOT_RESOLVED" in msg or "ERR_INTERNET_DISCONNECTED" in msg:
                print(f"[ERROR] Cannot connect to {url} (DNS / network issue)")
                print("Fix: check your network connection or the URL.")
            else:
                print(f"[ERROR] Failed to open {url}: {msg}")
            print("=" * 60)
            try:
                self.driver.quit()
            except Exception:
                pass
            sys.exit(1)
    
    def full_screen(self):
        self.driver.maximize_window()
        pyautogui.hotkey("f11")

    def open_book_mark():
        pyautogui.hotkey("ctrl", "shift", "b")

    def switch_to_game_web(self):
        """Call this after the page opens a new tab, so selenium can control it."""
        print("switching web page")
        self.driver.switch_to.window(self.driver.window_handles[-1])

        