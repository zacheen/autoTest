import os
# os.system('python FKNN_selenium.py')

import threading
import time

filename = "FKNN_selenium.py"
with open(filename, "rb") as source_file :
    code = compile(source_file.read(), filename, "exec")

print("open many web")

class Open_web_thread (threading.Thread):   #继承父类threading.Thread
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数     
        print("open website")
        # !python FKNN_selenium.py
        # exec(code)
        os.system('python FKNN_selenium.py')

for x in range(100):
    open_web_thread = Open_web_thread()
    open_web_thread.start()
    time.sleep(30)
