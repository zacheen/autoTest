from util.Click import ClickPyautogui

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

import os
import sys
import time
import glob
# Same X-less guard as util/Click — see comment there. 
  # pyautogui here is only touched on the non-headless paths (window focus + F11 fullscreen)
  # headless path uses a fixed CDP viewport and never queries the screen.
try:
    import pyautogui
except Exception:
    pyautogui = None

# ── Browser mode toggles ──────────────────────────────────────────
# HEADLESS : True  → Chrome --headless=new, no visible window, viewport via CDP
#            False → real window, F11 fullscreen, needs a desktop
# use_sel  : 1     → ClickSelenium (ActionChains clicks, driver.get_screenshot_as_png)
#                    — coord system = browser viewport
#            0     → ClickPyautogui (OS-level clicks + pyautogui.screenshot)
#                    — coord system = OS screen, only makes sense with HEADLESS=False
# Typical combos:
#   HEADLESS=True  + use_sel=1  → containerized (Colab, CI, server)
#   HEADLESS=False + use_sel=1  → local dev, watch Chrome, browser-internal capture
#   HEADLESS=False + use_sel=0  → legacy OS-screen pipeline
HEADLESS = True
use_sel = 1

# DPR must match what pyautogui sees so screen-coord templates work via chromedriver.
# Single source of truth — used by both --force-device-scale-factor and the headless
# CDP viewport override below.
DEVICE_SCALE_FACTOR = 1.25

# Headless viewport is a FIXED logical size, independent of the host screen, so the
# page renders at identical pixel dimensions everywhere (local dev, CI, Colab) and
# the agent's screen-coord templates stay valid regardless of whose monitor runs it.
# Matches the Colab notebook's AUTOTEST_HEADLESS_VIEWPORT setting.
# Override per-run via AUTOTEST_HEADLESS_VIEWPORT="WxH".
HEADLESS_VIEWPORT = "1920x1080"

def _resolve_chromedriver():
    """Return a Service for chromedriver.

    Try online first: ChromeDriverManager().install() reaches Chrome-for-Testing
    to pick the version matching the installed Chrome, downloads it, and caches it
    under ~/.wdm. When offline that network call raises (ConnectionError), so we
    fall back to the newest chromedriver.exe already sitting in the ~/.wdm cache —
    i.e. whatever a previous online run downloaded. No manual file placement needed;
    just run online once to populate the cache.
    """
    try:
        return Service(ChromeDriverManager().install())
    except Exception as e:
        cache_root = os.path.join(os.path.expanduser("~"), ".wdm")
        candidates = glob.glob(
            os.path.join(cache_root, "**", "chromedriver.exe"), recursive=True
        )
        if candidates:
            # Pick the most recently downloaded driver in the cache.
            local = max(candidates, key=os.path.getmtime)
            print(f"[Chrome_Driver] online driver resolution failed ({e!r}); "
                  f"falling back to cached driver: {local}")
            return Service(local)
        print("=" * 60)
        print(f"[ERROR] Could not resolve chromedriver online ({e!r}) and no "
              f"cached driver was found under {cache_root}.")
        print("Fix: connect to the internet once so webdriver_manager can "
              "download and cache the matching chromedriver, then offline runs "
              "will reuse it automatically.")
        print("=" * 60)
        raise


class Chrome_Driver:
    def __init__(self, game_env):
        """Open browser, log in, and wire the driver into the session."""
        print("open browser")
        options = webdriver.ChromeOptions()
        if HEADLESS:
            # New headless mode (Chrome 109+). Renders to off-screen buffer; same
            # viewport size honoured, get_screenshot_as_png() still works.
            options.add_argument("--headless=new")
            # Colab / containerized Linux runs as root and ships a tiny /dev/shm.
            # --no-sandbox lets Chrome run as root; --disable-dev-shm-usage falls
            # back to /tmp for shared memory. Both are no-ops on local Windows.
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1960,1080")
        # Pin DPR so get_screenshot_as_png() pixels match the .txt region coords
        options.add_argument(f"--force-device-scale-factor={DEVICE_SCALE_FACTOR}")
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

        service = _resolve_chromedriver()
        self.driver = webdriver.Chrome(service=service, options=options)
        self.game_env = game_env

        if HEADLESS:
            # --window-size is unreliable in headless (Chrome defaults to ~800×600).
            # Force the CSS viewport via CDP to a FIXED size (HEADLESS_VIEWPORT),
            # not the host screen, so:
            #   - CSS width / height = HEADLESS_VIEWPORT / DPR
            #   - get_screenshot_as_png() returns physical pixels = HEADLESS_VIEWPORT
            # The page therefore renders identically on every machine (and matches
            # Colab), so existing screen-coord templates keep working. This also
            # means no pyautogui screen query is needed — safe in X-less envs.
            # AUTOTEST_HEADLESS_VIEWPORT="WxH" overrides the fixed default.
            viewport = os.environ.get("AUTOTEST_HEADLESS_VIEWPORT", HEADLESS_VIEWPORT)
            screen_w, screen_h = map(int, viewport.lower().split("x"))
            self.driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": int(screen_w / DEVICE_SCALE_FACTOR),
                "height": int(screen_h / DEVICE_SCALE_FACTOR),
                "deviceScaleFactor": DEVICE_SCALE_FACTOR,
                "mobile": False,
            })

        if not HEADLESS:
            # Focus the browser window so pyautogui hotkeys land in Chrome.
            # In headless mode there's no window to focus.
            ClickPyautogui.click(30, 30)
            time.sleep(1)
        self.full_screen()
        self.login_plat()

        # Diagnostic — confirm viewport and DPR are what we expect
        vw = self.driver.execute_script("return window.innerWidth")
        vh = self.driver.execute_script("return window.innerHeight")
        dpr = self.driver.execute_script("return window.devicePixelRatio")
        print(f"[Chrome_Driver] viewport: {vw}×{vh}, DPR: {dpr}")

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
        if not HEADLESS:
            # F11 toggles browser fullscreen. In headless mode there's no window,
            # and the global hotkey would land in whatever OS app has focus — bad.
            pyautogui.hotkey("f11")

    def open_book_mark():
        pyautogui.hotkey("ctrl", "shift", "b")

    def switch_to_game_web(self):
        """Call this after the page opens a new tab, so selenium can control it."""
        print("switching web page")
        self.driver.switch_to.window(self.driver.window_handles[-1])

        