from extendFunction import imageSearch
from extendFunction import clickImage
import time
import os, os.path

class imageSearchAndClickImage:
    def __init__(self, browser, picturePath, testDescription, errorMsg, className, screenShotSwitch):
        
        self.browser = browser
        self.picturePath = picturePath
        self.testDescription = testDescription
        self.errorMsg = errorMsg
        self.className = className
        self.screenShotSwitch = screenShotSwitch

    def imageSearchAndClickImage(self):
        while True:
                self.picPostiton = imageSearch.imageSearch( self.picturePath, 0.7)
                if self.picPostiton != [-1,-1]:
                    self.picNumber=time.strftime("%Y-%m-%M-%H_%M_%S",time.localtime(time.time()))
                    if self.screenShotSwitch == True:
                        self.browser.get_screenshot_as_file('./testreport/testpic/'+self.className+'.png')
                    time.sleep(0.5)
                    clickImage.clickImage(self.picPostiton, "left", 0.01)
                    time.sleep(0.1)
                    break