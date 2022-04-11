import pyautogui

class clickImage:
    def __init__(self, pos, action, timestamp, offset=5):
        self.pos = pos
        self.action = action
        self.timestamp = timestamp
        self.offset = offset

    def clickImage(self):
        pyautogui.moveTo( self.pos[0] + self.offset, self.pos[1] + self.offset, self.timestamp )
        pyautogui.click( button=self.action )