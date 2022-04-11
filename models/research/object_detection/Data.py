def get_name_list(label,index):
    try : 
        return name_list[label][index]
    except :
        # 其他的 數字本身即 name_list
        return index

# 填寫方式
# 首先要先新增各遊戲的dict
# EX :  name_list["FKNN"] = {}
#       class_list["FKNN"] = {}
# 再來填要辨識東西對應的list
# !!!記得第一格都是填 "沒截到" 或 "沒用到"!!! (辨識完的數字不會出現0 因此要先把list[0]補滿) 
#   name_list : 辨識出來的數字轉資料用的
#       EX : name_list["FKNN"]["bank_mul"] = ["沒截到","抢庄倍数 1","抢庄倍数 2","抢庄倍数 3","不抢庄"]  # 抢庄動作
#       通常後面的list裡面是各個辨識完的數字對應的資料
#   class_list : 辨識出來的數字轉label用的list (identify_for_import.py 的writ_xml如果=True 就會需要用到)
#       EX :  class_list["FKNN"]["playing_number"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"]
#       通常裡面的list順序會跟 training\labelmap.pbtxt 裡面一樣 

name_list = {}
class_list = {}

# 20210223 修改 class_list不分遊戲
class_list["bank_mul"] = ["沒截到","ch_1","ch_2","ch_3","ch_no"]
class_list["other_mul"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"] 
class_list["player_money_bef"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"]
class_list["player_money_aft"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"]
class_list["win_lose"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 
class_list["playing_number"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 
class_list["fin_card_suit"] = ["沒截到","spade", "heart", "diamond", "club"]

class_list["card_num"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]
class_list["card_num_12"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]
class_list["card_num_3"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]
class_list["card_num_4"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]
class_list["card_num_0"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]
class_list["card_num_5"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_10","num_11","num_12","num_13"]

class_list["card_suit"] = ["沒截到","spade","heart","club","diamond"]
class_list["card_suit_12"] = ["沒截到","spade","heart","club","diamond"]
class_list["card_suit_34"] = ["沒截到","spade","heart","club","diamond"]
class_list["card_suit_05"] = ["沒截到","spade","heart","club","diamond"]
class_list["ma_jang_end"] = ["沒截到","ma_1W","ma_2W","ma_3W","ma_4W","ma_5W","ma_6W","ma_7W","ma_8W","ma_9W","ma_EAST","ma_SOUTH","ma_WEST","ma_NORTH","ma_BAI","ma_FA","ma_CHONG"]
class_list["ma_jang"] = ["沒截到","ma_1W","ma_2W","ma_3W","ma_4W","ma_5W","ma_6W","ma_7W","ma_8W","ma_9W","ma_EAST","ma_SOUTH","ma_WEST","ma_NORTH","ma_BAI","ma_FA","ma_CHONG"]

name_list["FKNN"] = {}
name_list["FKNN"]["bank_mul"] = ["沒截到","抢庄倍数 1","抢庄倍数 2","抢庄倍数 3","不抢庄"]  # 抢庄動作
name_list["FKNN"]["ohter_mul"] = ["沒截到"]  # 下注

# identify_a_thing 裡要寫成 xml 時在用的
# class_list["FKNN"] = {}
# class_list["FKNN"]["bank_mul"] = ["沒截到","ch_1","ch_2","ch_3","ch_no"]
# class_list["FKNN"]["other_mul"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"] 
# class_list["FKNN"]["win_lose"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 
# class_list["FKNN"]["playing_number"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 
# class_list["FKNN"]["fin_card_suit"] = ["沒截到","spade", "heart", "diamond", "club"]

name_list["KPQZNN"] = {}
# class_list["KPQZNN"] = {}
# class_list["KPQZNN"]["playing_number"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 
# class_list["KPQZNN"]["player_money_aft"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"] 
# class_list["KPQZNN"]["player_money_bef"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0"] 


name_list["HHDZ"] = {}
# class_list["HHDZ"] = {}

name_list["WXHH"] = {}
# class_list["WXHH"] = {}

name_list["QZNN"] = {}
# class_list["QZNN"] = {}
# class_list["QZNN"]["playing_number"] = ["沒截到","num_1","num_2","num_3","num_4","num_5","num_6","num_7","num_8","num_9","num_0","plus","neg"] 



name_list["D21D"] = {}








