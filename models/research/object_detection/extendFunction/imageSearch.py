import pyautogui
import cv2
import numpy as np

class imageSearch:
    def __init__(self, image, precision = 0.7):
        self.image = image
        self.precision = precision

    def imageSearch(self):
        self.referenceImage = pyautogui.screenshot()
        self.img_rgb = np.array(self.referenceImage)
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        self.template = cv2.imread(self.image, 0)
        self.template.shape[::-1]

        self.res = cv2.matchTemplate(self.img_gray, self.template, cv2.TM_CCOEFF_NORMED)
        self.min_val, self.max_val, self.min_loc, self.max_loc = cv2.minMaxLoc(self.res)
        if self.max_val < self.precision:
            return [-1,-1]
        return self.max_loc #返回圖片座標