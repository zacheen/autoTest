import pyautogui
import time
from selenium.webdriver.common.action_chains import ActionChains

# where the cursor parks after each click (top-middle of screen)
HOME_POS = (952, 21)

# Click the screen using pyautogui or selenium
def get_ctrl(driver=None):
    if driver:
        return ClickSelenium(driver)
    return ClickPyautogui

class ClickAbstract:
    """Abstract pointer-input backend. Subclasses: ClickPyautogui, ClickSelenium."""
    # Subclasses override to shave off sleep that ActionChains already spends
    sleep_adjust = 0.0

    @staticmethod
    def click(x, y, long_click=None, move_click=None):
        '''
        move_click :
            move to (x,y)
            wait move_click seconds
            and click
        long_click :
            mouse down at (x,y)
            wait long_click seconds
            mouse up
        '''
        raise NotImplementedError

    @staticmethod
    def drag_swipe(direction, times):
        raise NotImplementedError

    @staticmethod
    def move_home():
        """Park the cursor somewhere neutral. Optional (selenium has no real cursor)."""
        pass

class ClickPyautogui(ClickAbstract):
    """Screen-coordinate input via pyautogui — works with any window."""
    @staticmethod
    def click(x, y, long_click=None, move_click=None):
        if long_click is not None:
            pyautogui.mouseDown(x, y)
            time.sleep(long_click)
            pyautogui.mouseUp()
        else :
            if move_click is not None:
                pyautogui.moveTo(x, y)
                time.sleep(move_click)
            pyautogui.click(x, y)
        ClickPyautogui.move_home()

    @staticmethod
    def drag_swipe(direction, times):
        for _ in range(times):
            if direction == "left":
                pyautogui.mouseDown(400, 600)
                pyautogui.moveTo(1500, 600, 1.5)
                time.sleep(0.5)
                pyautogui.mouseUp()
                time.sleep(0.5)
            elif direction == "right":
                pyautogui.mouseDown(1500, 600)
                pyautogui.moveTo(400, 600, 1.5)
                time.sleep(0.4)
                pyautogui.mouseUp()
                time.sleep(0.2)
        ClickPyautogui.move_home()

    @staticmethod
    def move_home():
        pyautogui.moveTo(*HOME_POS)


class ClickSelenium(ClickAbstract):
    """Page-coordinate input via selenium ActionChains.
    Note: long_click / move_click are pyautogui-only — silently ignored here."""

    sleep_adjust = -0.7  # ActionChains already waits inside perform()

    def __init__(self, driver):
        self.driver = driver

    def click(self, x, y, long_click=None, move_click=None):
        (ActionChains(self.driver)
            .move_by_offset(x, y)
            .click()
            .move_by_offset(-x, -y)
            .perform())

    def drag_swipe(self, direction, times):
        for _ in range(times):
            if direction == "left":
                self._hold_and_drag(start=(400, 600), step=(55, 0),
                                    steps=20, reset=(-1500, -600))
            elif direction == "right":
                self._hold_and_drag(start=(1500, 600), step=(-55, 0),
                                    steps=20, reset=(-400, -600),
                                    re_center_around_release=True)

    def _hold_and_drag(self, start, step, steps, reset,
                       re_center_around_release=False):
        ActionChains(self.driver).move_by_offset(*start).click_and_hold().perform()
        for _ in range(steps):
            ActionChains(self.driver).move_by_offset(*step).perform()
        if re_center_around_release:
            ActionChains(self.driver).move_by_offset(0, 0).perform()
        ActionChains(self.driver).release().perform()
        if re_center_around_release:
            ActionChains(self.driver).move_by_offset(0, 0).perform()
        ActionChains(self.driver).move_by_offset(*reset).perform()
