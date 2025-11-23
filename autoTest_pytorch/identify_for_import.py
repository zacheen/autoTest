# label 格式
# 
# num_1  num_2  num_3  num_4  num_5  num_6  num_7  num_8  num_9  num_0  plus  neg   ch_no
#     1      2      3      4      5      6      7      8      9     10    11   12   13
#  ch_1   ch_2   ch_3   ch_4   ch_5   ch_6   ch_7   ch_8   ch_9   ch_0              不     

# 鬼, 黑, 紅, 梅, 方
#  1, 2,  3,  4,  5,

#       1        2        3        4        5        6        7        8        9        10        11        12        13
#       A        2        3        4        5        6        7        8        9        10         J         Q         K
#  card_1   card_2   card_3   card_4   card_5   card_6   card_7   card_8   card_9   card_10   card_11   card_12   card_13


import os
# 這個要放在 import tensorflow as tf 之前
# 決定要不要顯示warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
# import tensorflow as tf
import sys
import time

# 把上一層目錄加到環境變數
sys.path.append("..")

# # Import utilites
# from utils import label_map_util
# from utils import visualization_utils as vis_util

# glob 是用來找資料夾路徑的
import glob
from pathlib import Path

import datetime
ISOTIMEFORMAT = '%Y_%m_%d_%H_%M_%S' 

# 宣告targetPattern 裡面存的是規律 
# 如果符合這個規律的檔案才會進入辨識
# EX: 下面的 file_name_end
targetPattern = None
# 這個是規律的一部份
# 尾端結尾要是 ".PNG" 才符合規律
file_name_end = ".PNG"

# write_to_xml 用的文字檔
part_1 = r"""<annotation>
	<folder>pic</folder>
    """

part_2 = r"""</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		"""
part_3 = r"""		<depth>3</depth>
	</size>
	<segmented>0</segmented>"""

loop_txt = r"""</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>"""

#  !!!!!!!!!!!!最好只 import 各別遊戲(不過還沒實做) (好像也還好??)
# import Data

# 這個變數應該要像 Tool_Main.server_using 一樣 交給Thread處理
# 使用中 = True ， 使用完畢 = False
iden_using = False
game_name = None

last_MODEL_NAME = None

# 初始化 要用來辨識的各個變數
# 這些參數都是從 identify_number, identify_a_thing, identify_string 後面() 裡面的變數直接帶過來的
def initial(iden_thing, round_count = None) : 
    global last_MODEL_NAME

    # -----------------------------------------------------------------------------------------------
    # 與本次辨識圖片路徑有關
    # 取得執行時的目錄
    CWD_PATH = os.getcwd()
    # 取得 user_change 資料夾路徑
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent
    user_change_path = list(parent_dir.rglob("user_change"))[0]
    # 做出要被辨識圖片的路徑
    absolute_dst = user_change_path / "game_pic" / f"{game_name} + _pic" / "iden_thing"

    # 做出規律(符合這個規律底下的檔案才會被辨識)
    global targetPattern
    if round_count == None:
        targetPattern = (absolute_dst / "*").with_suffix(file_name_end)
    else:
        targetPattern = (absolute_dst + f"*_{round_count}").with_suffix(file_name_end)
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # 以下與讀取model有關
    # 要讀入的model資料夾路徑
    MODEL_NAME = str(game_name)+"_"+iden_thing+"_inference_graph"
    # 路徑到frozen detection graph .pb file 這個檔案就是要讀入的model
    PATH_TO_CKPT = os.path.join(CWD_PATH,"training_model",MODEL_NAME,'frozen_inference_graph.pb')
    # print(PATH_TO_CKPT)
    #檢查有沒有 inference_graph 的資料夾 而且是個遊戲獨自model的資料夾
    if os.path.isfile(PATH_TO_CKPT) == False : 
        # 同上只是檢查有沒有預設的model
        MODEL_NAME = "ALL_"+iden_thing+"_inference_graph"
        PATH_TO_CKPT = os.path.join(CWD_PATH,"training_model",MODEL_NAME,'frozen_inference_graph.pb')
        # print(PATH_TO_CKPT)
        if os.path.isfile(PATH_TO_CKPT) == False : 
            print("inference_graph 資料夾 還沒改名!!!")
            raise Exception
        else:
            print(iden_thing+"使用共用model")
    else:
        print(iden_thing+"使用"+game_name+"專用model")

    if last_MODEL_NAME == MODEL_NAME :
        return
    else:
        last_MODEL_NAME = MODEL_NAME

    global sess
    # 開始讀 model 
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    global image_tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    global detection_boxes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    global detection_scores
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    global detection_classes
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    global num_detections
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    

# 單純把 label 轉成 數字回傳 
# (用在要辨識的東西只單純包含數字)
def class_to_num(this_class):
    # this_class should be number
    if this_class >= 1 and this_class <= 9 : 
        return this_class
    else :
        return 0

# 用來儲存 辨識結果
# num : label
# box : 這個label出現的位置
class Num_place():
    def __init__(self,num,box):
        self.num = num
        self.box = box

    def __str__(self) :
        return "num : "+str(self.num)+" ,box : "+str(self.box) 

    def __repr__(self) :
        return str(self)

# 找到 這個 box 正中間的位置
# (給 find_pos_array 用的)
def find_mid(box): 
    # -----------
    # |    -> X
    # | |
    # | ˇ
    # | Y
    ymin, xmin, ymax, xmax = box
    
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2

    return (x , y)   

# 判斷 this_box 應該要放在 num_array 哪個位置
# return (第一個, 第二個)
#     第一個 : 可以塞的位置(第二個為True) 或 哪個位置已經有東西了(第二個為False)
#     第二個 : 可不可以插入
#           False : 代表 這個位置已經有東西了 
def find_pos_array(this_box, num_array):
    # 如果這個list還沒有東西 一定可以加進去
    if len(num_array) == 0:
        return (0, True)
    
    # 先取得中間位置
    x,y = find_mid(this_box)
    # print(this_box,x,y)

    for i in range(len(num_array)) :
        ymin, xmin, ymax, xmax = num_array[i].box
        

        # 怕xy搞混 先做紀錄 (筆記中圈起來5)
        # -----------
        # |    -> X
        # | |
        # | ˇ
        # | Y
        # 代表找到測試的那個位置了
        if x > xmin :
            if x < xmax :
                # 代表目前傳進來的東西的位置，在此張圖片裡面
                return (i, False)
            if x > xmax :
                # 代表位置還沒到
                continue
        elif x < xmin :
            # 代表他跳過一個位置
            # 因此傳進來的東西可以塞在這個位置

            return (i, True)     

    # 全部的位置都找過了(代表比過最後一個後 還是沒有在裡面)  
    # 因此可以塞在最後面

    return (len(num_array), True)

def cal_area(box):
    ymin, xmin, ymax, xmax = box
    return (xmax-xmin)*(ymax-ymin)

# box_b 要放比較右邊的
def cal_iou(box_a,box_b):
    # print(box_a,box_b)
    # 初步檢查(算是優化)
    # 如果box_b是在右邊 代表這兩個box x 邊完全沒重疊
    # print("in cal_iou:",box_a, box_b)
    # if box_b[1] > box_a[3] :
    #     return 0
    
    a_ymin, a_xmin, a_ymax, a_xmax = box_a
    b_ymin, b_xmin, b_ymax, b_xmax = box_b

    # 尋找重疊的邊
    x_min = max(a_xmin, b_xmin)
    x_max = min(a_xmax, b_xmax)
    # 重疊的邊有多大
    x_len = x_max-x_min
    # print("x_len : ",x_len)
    # 沒重疊 直接回0
    if x_len <= 0 :
        return 0

    y_min = max(a_ymin, b_ymin)
    y_max = min(a_ymax, b_ymax)
    # 重疊的邊有多大
    y_len = y_max-y_min
    # print("y_len : ",y_len)
    # 沒重疊 直接回0
    if y_len <= 0 :
        return 0

    cover_area = x_len*y_len*1.0

    #找面積小的那個框
    box_a_area = cal_area(box_a)
    box_b_area = cal_area(box_b)
    area_min = min(box_a_area, box_b_area)

    return cover_area/area_min

# check_num : 要檢查相鄰幾個沒有重疊
def find_pos_array_iou(this_box, num_array, iou = 0.7, check_num = 2):    
    # 如果這個list還沒有東西 一定可以加進去
    array_len = len(num_array)
    if array_len == 0:
        return (0, True)
    
    # 取得最左邊位置
    this_ymin, this_xmin, this_ymax, this_xmax = this_box

    for i in range(array_len) :
        ymin, xmin, ymax, xmax = num_array[i].box
        # 怕xy搞混 先做紀錄 (筆記中圈起來5)
        # -----------
        # |    -> X
        # | |
        # | ˇ
        # | Y
        # 代表找到測試的那個位置了
        if this_xmin > xmin :
            continue
        elif this_xmin < xmin :
            # 代表他跳過一個位置
            # 因此傳進來的東西可以塞在這個位置
            
            # 左邊的
            for ii in range(max((i-check_num+1),0),i+1):
                # 如果 iou 大於設定值 代表重疊面積太高 代表是同一個東西
                ret_iou = cal_iou(num_array[ii].box, this_box)
                # print(ret_iou)
                if ret_iou > iou :
                    return (i, False)
            # 右邊的
            for ii in range(i+1,min((i+check_num+1),array_len)):
                # 如果 iou 大於設定值 代表重疊面積太高 代表是同一個東西
                ret_iou = cal_iou(this_box, num_array[ii].box)
                # print(ret_iou)
                if ret_iou > iou :
                    return (i, False)

            return (i, True)     

    # 全部的位置都找過了(代表比過最後一個後 還是沒有在裡面)  
    # 因此可以塞在最後面
    # 左邊的
    for ii in range(max((i-check_num+1),0),array_len):
        # 如果 iou 大於設定值 代表重疊面積太高 代表是同一個東西
        ret_iou = cal_iou(num_array[ii].box, this_box)
        # print(ret_iou)
        if ret_iou > iou :
            return (None, False)
            
    return (array_len, True)

# 拿來查看辨識結果 (或是把結果寫成可以再放入training的資料格式)
# 前提是要在 Data.py 中填 label的list(如何填我寫在 Data.py 中)
# iden_thing 用來取 Data.py 中 用哪一個list 轉換成 label (就是丟進 cut_pic_data location 的值)
# file_name : pic 檔案的名字
# img : 已經打開的 file_name 打開的 img
# num_array : 圖片辨識的結果
# 這個裡面就是單純的處理字串因此就不寫注解了
def write_to_xml(iden_thing, file_name, img, num_array) : 
    w_dst = file_name.replace(file_name_end,".xml")
    with open(w_dst, "w") as write_dst_f : 
        #寫開頭
        write_dst_f.write(part_1)

        #寫檔名 那一行
        write_dst_f.write("<filename>"+ file_name.split("\\")[-1] +"</filename>\n")  # 只要檔名就好 不要路徑
        
        #寫開頭中間
        write_dst_f.write("""	<path>""")
        write_dst_f.write(file_name)
        write_dst_f.write(part_2)

        #寫高與寬
        size = img.shape
        height, width, channel = size
        # print(size)

        write_dst_f.write("<width>"+ str(width) +"</width>\n")
        write_dst_f.write("\t\t<height>"+ str(height) +"</height>\n")

        #繼續寫完開頭
        write_dst_f.write(part_3)
        
        for pos_and_class in num_array : 
            #寫for開頭
            write_dst_f.write("\n\t<object>\n")
            class_name = Data.class_list[iden_thing][pos_and_class.num]
            # class_name = pos_and_class
            write_dst_f.write("\t\t<name>" + str(class_name))
            write_dst_f.write(loop_txt)
            
            ymin, xmin, ymax, xmax = pos_and_class.box
            # print("ymin : " + str(ymin))
            # print("xmin : " + str(xmin))
            # print("ymax : " + str(ymax))
            # print("xmax : " + str(xmax))
            xmin = xmin * width
            xmax = xmax * width
            ymin = ymin * height
            ymax = ymax * height

            write_dst_f.write(str(int(xmin)) + "</xmin>\n\t\t\t<ymin>")
            write_dst_f.write(str(int(ymin)) + "</ymin>\n\t\t\t<xmax>")
            write_dst_f.write(str(int(xmax)+1) + "</xmax>\n\t\t\t<ymax>" +str (int(ymax)+1)+"</ymax>\n")

            # 寫 loop 結尾
            write_dst_f.write("\t\t</bndbox>\n\t</object>")

        #結尾
        write_dst_f.write("\n</annotation>")

# 把單純不重複位置的東西全部按照位置順序抓出來
# (要會用這個 !!!)
# 把辨識結果傳進來 boxes, classes, scores
# 還有 min_score_thresh 嚴格程度
def box_to_thing_list(
    boxes,
    classes,
    scores,
    min_score_thresh = 0.6,
    iou = 0.7
    ) :

    # num_array 回傳結果
    num_array = []

    count = 0

    # 按框框不重複(使用find_pos_array檢查)，擺入結果(從最像的開始擺) 
    for i in range(boxes.shape[0]) :
        # print("想要放 :",classes[i], "score:", scores[i])
        if scores[i] < min_score_thresh :
            # 如果分數已經小於最小可接受範圍(min_score_thresh) 則跳出
            break

        # 尋找可以放的位置
        position, space = find_pos_array_iou(boxes[i],num_array,iou)
        
        # 如果可以放進去，就把結果放進去
        if space :
            # print(classes[i],boxes[i],scores[i],position)
            num_array.insert(position, Num_place(classes[i],boxes[i]))
            # print(num_array)

        # 以下4行為測是用 以後要刪掉的
        # if count < 30 :
        #     print(classes[i],boxes[i],scores[i],position)
        #     num_array.append(Num_place(classes[i],boxes[i]))
        #     count += 1

    # print(num_array)
    return num_array

def all_in(
    boxes,
    classes,
    scores,
    min_score_thresh = 0.6
    ) :

    # num_array 回傳結果
    num_array = []

    count = 0

    # 按框框不重複(使用find_pos_array檢查)，擺入結果(從最像的開始擺) 
    for i in range(boxes.shape[0]) :
        if scores[i] < min_score_thresh :
            # 如果分數已經小於最小可接受範圍(min_score_thresh) 則跳出
            break

        if count < 10 :
            # print(classes[i],boxes[i],scores[i])
            num_array.append(Num_place(classes[i],boxes[i]))
            count += 1

    # print(num_array)
    return num_array

# num_array : 辨識結果(label)的list (經過 box_to_XXX 的回傳結果)
# class_to_str : 每一個辨識結果對應要變成的字串 (identify_string 裡 class_to_str 傳進來的參數)
# (要會用這個 !!!)
def list_to_str(num_array, class_to_str):
    return_str = ""

    for this_num in num_array :
        # 把辨識結果 轉換成 對應的字 並相加
        return_str += class_to_str[this_num.num]

    return return_str

# num_array : 辨識結果(label)的list (經過 box_to_XXX 的回傳結果)
# class_to_list : 每一個辨識結果對應要變成的字串 (identify_string 裡 class_to_str 傳進來的參數)
# (要會用這個 !!!)
def list_to_info_list(num_array, class_to_list):
    return_str = []

    if class_to_list == None :
        for this_num in num_array :
            # 把辨識結果 轉換成 對應的字 並相加
            return_str.append(this_num.num)
    else :
        for this_num in num_array :
            # 把辨識結果 轉換成 對應的字 並相加
            return_str.append(class_to_list[this_num.num])

    return return_str

# 把辨識的結果傳進來
# 會回傳 從圖片左到右 不重複位置的結果 : Num_place 的 list
# 且 + - 號只能出現在第一個位置(如果有的話)
# (要會用這個 !!!)
# box_to_thing_list的進階版(因為有判斷+-號可以出現的位置)
def box_to_num_list(
    boxes,
    classes,
    scores,
    min_score_thresh = 0.6
    ):

    # num_array 回傳結果
    num_array = []
    neg_or_plus = False

    # 1. 找到最像的 +- 後擺第一個位置
    # 2. 按框框不重複，擺入結果(從最像的開始擺) 
    for i in range(boxes.shape[0]) :
        if scores[i] < min_score_thresh :
            # 如果分數已經小於最小可接受範圍(min_score_thresh) 則跳出
            break
        
        # 如果是 +- 則強制擺入
        if (classes[i] == 11 or classes[i] == 12) :
            if neg_or_plus == False :
                # 因為他一定是第一個 所以直接加 而且如果有位置重疊 去掉第一個 再寫入
                neg_or_plus = True
                position, space = find_pos_array(boxes[i],num_array)
                # print("find neg_or_plus at : "+str(position))
                if space == False and position == 0:
                    #代表 plus or neg 的位置已經被 占走了
                    print("第一個位置已經有數字 : "+str(num_array[0].num)+" 直接替換成"+str(classes[i]))
                    num_array[0] = Num_place(classes[i],boxes[i])        

                elif space == False and position != 0 :
                    # 在其他位置找到 有問題 !  (我現在先PASS 找下一個)
                    neg_or_plus = False

                else :
                    # 正常情況 ( +- 號位置沒有東西 且 在最前面)
                    num_array.insert(0, Num_place(classes[i],boxes[i]))
        # 如果是普通數字
        else:
            position, space = find_pos_array(boxes[i],num_array)
            if space :
                num_array.insert(position, Num_place(classes[i],boxes[i]))
                # print(num_array)

    return num_array

# 如果單純只有數字和符號 可以帶入Num_place 的 list (box_to_num_list的回傳值)
# 回傳 int EX: 123456
def list_to_num(num_array):
    total_num = 0
    # 控制最後的正負號
    pos_or_neg = 1
    for this_num in num_array :
        # 文字轉數字
        if this_num.num >= 1 and this_num.num <= 10 : 
            total_num = total_num*10 + class_to_num(this_num.num)
        elif this_num.num == 11 :
            # 如果是正號
            pass
        elif this_num.num == 12 :
            # 如果是負號
            # neg
            pos_or_neg = -1
        else:
            print(this_num,"已超出數字範圍!!")
            raise Exception

    return total_num * pos_or_neg


# identify_XXX 的關鍵用法
# 初始化 神經網路
# 把檔名為iden_thing的資料夾裡面的檔案傳回list中
# for 資料夾裡面的檔案(要辨識的檔案)
#      辨識
#      把辨識的結果 變成 list
#      處理這個list #數字、字串、物體在這個階段的處理方式不同(差異在這)，因此作法不同，才會有三種func
#      把處理完後的結果加回return_info的list


# 通常拿來辨識單純的數字 EX: 攜帶分數 、 "下幾倍"(如果只辨識數字)
# 如果圖中一個數字都沒有會回傳 0 (如果辨識訓練的好的話)
# 會回傳這個資料夾裡面的圖的辨識結果字串 EX: [123,-456,456789]

# iden_thing : 要辨識的分類 (就是丟進 cut_pic_data location 的值)
#              會去找 C:\python\object_detection\user_change\ 底下 + 各自遊戲名稱資料夾 + iden_thing 裡面的圖片 
# round_count 才知道要找第幾round的圖片
# writ_xml 如果是True 會順便把辨識的結果 寫成可以訓練的 xml，先忽略
# thresh 辨識神經 所使用的域值 0~1 愈接近1愈嚴謹、嚴格 == 回傳的資料愈少
def identify_number(iden_thing, round_count, writ_xml = False, thresh = 0.5) : 
    #iden_thing -- C:\python\Thomas\object_detection\user_change\WXHH_pic\find_num→只需要放find_num
    # 像 server using 一樣，如果要開始執行之前 要把這個把開，代表現在有人在使用 
    global iden_using #判斷神經網路是否正在使用
    global targetPattern
    while iden_using :
        print("辨識",iden_thing, "等待中")
        time.sleep(2)
    iden_using = True
    
    # 初始化 辨識這個東西 所需要的參數 (EX: 讀取model)
    initial(iden_thing, round_count) #初始化神經網路
    # 把符合 targetPattern 的檔案列舉出來
    all_file = glob.glob(targetPattern) #帶入指定資料夾內符合檔名定義之欲辨識圖片檔案
    
    # return_info 裡面裝各個玩家的分數
    return_info = []
    # x 是為了輸出使用
    x = 1
    for each_file in all_file : 
        image = cv2.imread(each_file) #讀取圖片檔案
        image_expanded = np.expand_dims(image, axis=0) #從二維陣列轉換成一維陣列

        # 進行辨識
        # boxes - 圖片位置
        # scores - 是 classes 這個類別的可能性
        # classes - label(或是叫做類別)
        # num 沒用到
        # a = [boxes, scores, classes, num] 可以想像成list，但tuple(boxes, scores, classes, num)內的變數為絕對位置(不能變動) ※參考觀念
        (boxes, scores, classes, num) = sess.run(  #辨識完的結果 已提供辨識參考pattern的前提下，假設有張圖寫300，那張圖辨識完成後，會給出300個結果(300當作預設值)，且為程式自動定義裡面可能是數字的範圍 BOX:程式認定的範圍 score:相似度(0~1) class:辨識結果
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # 按ctrl看box_to_num_list
        ans_num = box_to_num_list(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            min_score_thresh=thresh)

        # 如果 writ_xml == True 會寫檔(可用來查看結果)
        # 可按ctrl看write_to_xml
        if writ_xml and (len(ans_num) != 0) :
            write_to_xml(iden_thing, each_file, image, ans_num)

        # 按ctrl看list_to_num
        ans_num = list_to_num(ans_num)
        # 把結果加入回傳list
        return_info.append(ans_num)
        # print(thing_name+" 第" + str(x) +"個 : " + str(ans_num)) #debug過程可以使用
        # read_dst_f.write(str(x) + " : ")

        x = x + 1
    iden_using = False
    return return_info

# 會回傳iden_thing資料夾裡面 辨識的結果 list (list 的每一格只會有一個值)
#   EX:[1,2,1,3,1] 可能 1代表愛心 2代表梅花
# 且此值尚未轉換成資料
# 通常在Main的 thread中處理回傳的東西 ( EX : FKNN_Main 的 set_card() )

# iden_thing : 要辨識的分類 (就是丟進 cut_pic_data location 的值)
#              會去找 C:\python\object_detection\user_change\ 底下 + 各自遊戲名稱資料夾 + iden_thing 裡面的圖片 
# round_count 才知道要找第幾round的圖片
# writ_xml 如果是True 會順便把辨識的結果 寫成可以訓練的 xml
# thresh 辨識神經 所使用的域值 0~1 愈接近1愈嚴謹、嚴格 == 回傳的資料愈少
def identify_a_thing(iden_thing, round_count, writ_xml = False, thresh = 0.5) :
    global iden_using
    while iden_using :
        print(iden_thing, "等待中")
        time.sleep(2)
    iden_using = True
    
    initial(iden_thing, round_count) #初始化神經網路
    all_file = glob.glob(targetPattern) #帶入指定資料夾內符合檔名定義之欲辨識圖片檔案
    
    return_info = []
    x = 1
    for each_file in all_file : 
        # 讀取圖片
        image = cv2.imread(each_file)
        image_expanded = np.expand_dims(image, axis=0)
        
        # 進行辨識
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # 要大於這個最低標準 才算有找到結果
        # scores[0][0] 是最像的結果的成績 看此值有沒有大於最低標準
        if scores[0][0] > thresh :
            # 取得最像的結果的辨識數字
            ans_num = np.squeeze(classes).astype(np.int32)[0]
            # 加入回傳list中
            return_info.append(ans_num)
            # 要不要寫檔的開關
            # 可按ctrl看write_to_xml
            if writ_xml :
                # thing_array 因為這個東西只會判斷最好的那一個 所以只有一個元素
                thing_array = []
                thing_array.insert(0, Num_place(ans_num, np.squeeze(boxes)[0]))
                write_to_xml(iden_thing, each_file, image, thing_array)
        # 如果成績太低則回傳None 代表沒找到東西
        else :
            return_info.append(None)
        # print(thing_name + " 為 : " + str(return_info))
        x = x + 1

    iden_using = False
    return return_info

# 辨識 要辨識的東西 是一個字串
# 會回傳這個資料夾裡面的圖的辨識結果字串 
#   EX: ["牛一","五花牛","牛牛","牛二"]
#       ["50-456789-456789"]

# iden_thing : 要辨識的分類 (就是丟進 cut_pic_data location 的值)
#              會去找 C:\python\object_detection\user_change\ 底下 + 各自遊戲名稱資料夾 + iden_thing 裡面的圖片 
# class_to_str EX: playing_number_list = ["沒截到",'1','2','3','4','5','6','7','8','9','0','+','-']
#              training 的時候是 字轉 label // 這裡是 label 轉字
# round_count 才知道要找第幾round的圖片
# writ_xml 如果是True 會順便把辨識的結果 寫成可以訓練的 xml
# thresh 辨識神經 所使用的域值 0~1 愈接近1愈嚴謹、嚴格 == 回傳的資料愈少
def identify_string(iden_thing, round_count, class_to_str, writ_xml = False, thresh = 0.5) : 
    # 像 server using 一樣，如果要開始執行之前 要把這個把開，代表現在有人在使用 
    global iden_using #判斷神經網路是否正在使用
    while iden_using :
        print(iden_thing, "等待中")
        time.sleep(2)
    iden_using = True
    
    # 初始化 辨識這個東西 所需要的參數 (EX: 讀取model)
    initial(iden_thing, round_count)
    
    # 把符合 targetPattern 的檔案列舉出來
    all_file = glob.glob(targetPattern) 
    
    # return_info 裡面裝各個玩家的辨識結果
    return_info = []
    
    # x 是為了輸出使用
    x = 1
    for each_file in all_file : 
        # 讀取圖片
        image = cv2.imread(each_file)
        image_expanded = np.expand_dims(image, axis=0)

        # 進行辨識
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # 按ctrl看box_to_thing_list
        ans_num = box_to_thing_list(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            min_score_thresh=thresh)

        # 如果 writ_xml == True 會寫檔(可用來查看結果)
        # 可按ctrl看write_to_xml
        if writ_xml and (len(ans_num) != 0) :
            write_to_xml(iden_thing, each_file, image, ans_num)

        # 按ctrl看list_to_str
        ans_num = list_to_str(ans_num, class_to_str)
        #["沒截到",'一','二','三','四','五','六','七','八','九','牛','没','小','炸弹','花']

        # 把結果加入 return_info 的list 中
        return_info.append(ans_num)
        print(iden_thing+" 第" + str(x) +"個 : " + str(ans_num))
        # read_dst_f.write(str(x) + " : ")

        x = x + 1

    iden_using = False
    return return_info

# class_to_info_list == None 直接回傳 id
def identify_things(iden_thing, round_count, class_to_info_list = None, writ_xml = False, thresh = 0.5, all_in_flag = False, iou = 0.7) : 
    # 像 server using 一樣，如果要開始執行之前 要把這個把開，代表現在有人在使用 
    global iden_using #判斷神經網路是否正在使用
    while iden_using :
        print(iden_thing, "等待中")
        time.sleep(2)
    iden_using = True
    
    # 初始化 辨識這個東西 所需要的參數 (EX: 讀取model)
    initial(iden_thing, round_count)
    
    # 把符合 targetPattern 的檔案列舉出來
    all_file = glob.glob(targetPattern) 
    
    # return_info 裡面裝各個玩家的辨識結果
    return_info = []
    
    # x 是為了輸出使用
    x = 1
    for each_file in all_file : 
        print(each_file)
        # 讀取圖片
        image = cv2.imread(each_file)
        image_expanded = np.expand_dims(image, axis=0)

        # 進行辨識
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # 按ctrl看box_to_thing_list
        # if all_in_flag :
        #     ans_num = all_in(
        #         np.squeeze(boxes),
        #         np.squeeze(classes).astype(np.int32),
        #         np.squeeze(scores),
        #         min_score_thresh=thresh)
        # else :
        ans_num = box_to_thing_list(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            min_score_thresh=thresh, 
            iou = iou)

        # 如果 writ_xml == True 會寫檔(可用來查看結果)
        # 可按ctrl看write_to_xml
        if writ_xml and (len(ans_num) != 0) :
            write_to_xml(iden_thing, each_file, image, ans_num)

        # 按ctrl看list_to_str
        ans_num = list_to_info_list(ans_num, class_to_info_list)
        #["沒截到",'一','二','三','四','五','六','七','八','九','牛','没','小','炸弹','花']

        # 把結果加入 return_info 的list 中
        return_info.append(ans_num)
        print(iden_thing+" 第" + str(x) +"個 : " + str(ans_num))
        # read_dst_f.write(str(x) + " : ")

        x = x + 1

    iden_using = False
    return return_info

if __name__ == '__main__' :
    print("start  __name__ == '__main__'")
    game_name = "ERMJ"
    print(game_name)

    # pass_data = identify_number("player_money_bef", None, writ_xml = True, thresh = 0.6)
    # playing_number_list = ["沒截到",'1','2','3','4','5','6','7','8','9','0',"+","-"]
    # pass_data = identify_number("win_lose", None, writ_xml = True, thresh = 0.6)

    # playing_number_list = ["沒截到",'1','2','3','4','5','6','7','8','9','0',"+","-"]
    # pass_data = identify_string("playing_number", None, playing_number_list, writ_xml = True, thresh = 0.6)

    # card_list = ["沒截到",'A','2','3','4','5','6','7','8','9','10',"J","Q","K"]
    # card_list = None
    # pass_data = identify_things("card_num_4", None, card_list, writ_xml = True, thresh = 0.5, all_in_flag = True)

    # card_list = None
    # pass_data = identify_things("card_suit_12", None, card_list, writ_xml = True, thresh = 0.5, all_in_flag = True)

    initial("ma_jang")
    print("初始化成功")

    card_list = ["沒截到",1,2,3,4,5,6,7,8,9,60,70,80,90,100,110,120]
    pass_data = identify_things("ma_jang", None, card_list, writ_xml = True, thresh = 0.3, all_in_flag = True, iou = 0.5)
    print(pass_data)

    # pass_data = identify_things("ma_jang", 1, card_list, writ_xml = True, thresh = 0.3, all_in_flag = True, iou = 0.5)
    # print(pass_data)
    
    # pass_data = identify_things("ma_jang", 2, card_list, writ_xml = True, thresh = 0.3, all_in_flag = True, iou = 0.5)
    # print(pass_data)

    # pass_data = identify_things("ma_jang", None, card_list, writ_xml = True, thresh = 0.5, all_in_flag = True, iou = 0.5)
    # print(pass_data)

    # pass_data = identify_things("ma_jang_end", None, card_list, writ_xml = True, thresh = 0.5, all_in_flag = True, iou = 0.5)
    # print(pass_data)

    # for x in range(0,1):  # 這個原本是用來填入 round_count 的
        # playing_number_list = ["沒截到",'1','2','3','4','5','6','7','8','9','0','+','-']
        # pass_data = identify_string("playing_number", None, playing_number_list, writ_xml = True, thresh = 0.6)
        # print(pass_data)

