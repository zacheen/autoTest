"""
用來做數學方面的計算與比較 (float處理)
float_to_int_100(float_num) : 經過特別處理的float 轉 int
compare_num(num1, num2) : 經過特別處理的比較兩數字
"""
import time
# import types
import numpy as np

 
# 經過特別處理的float(不管是字串或float型態) 轉 int
# float_num 要轉換的數字
# divide_num 後台小數點後面有幾位就要代幾個 0 EX: 99123.45 - 100 , 99.12345 - 100000
# 回傳為一個整數
def float_to_int_100(float_num, divide_num = 100000):
    # divide_num = 100000 #除1000的環境
    # divide_num = 100 #原本環境

    # 如果原本就是整數 就直接回傳回去
    if type(float_num) == type(1) :
        return float_num
    
    # 把數字轉成 float
    if type(float_num) == type("XXX.XX") :
        float_num = float(float_num)
    elif type(float_num) == type(1.0) :
        pass
    elif type(float_num) == type(np.float64(1.0))  :
        float_num = float(float_num)
    else:
        # 其他的型態就是錯誤的
        print("wrong type !!!!!!!! the type of "+ str(float_num) + " is : " + str(type(float_num)))
        raise Exception
    
    
    float_num = float_num*divide_num
    # 先把 float 轉成 str 再進行處理
    str_float = str(float_num)
    
    #假設 如果有小數點以下第三位 就是出現 xx.00000001 或 xx.99999999 的情況
    # print(str_float)
    # print(dot_place)
    dot_place = str_float.index(".")
    if len(str_float) == dot_place+2:
        # 代表格式 長這樣 : 12345.0  一定是整數 (小數點是倒數第二個)
        return int(float_num)
    # 其他的代表有問題
    elif str_float[dot_place+1]=="0" and str_float[dot_place+2]=="0" :
        # 處理 xx.00000001 這種
        # 很簡單 無條件捨去即可
        turn_int = int(float_num)
        return turn_int
    elif str_float[dot_place+1]=="9" and str_float[dot_place+2]=="9" :
        # 處理 xx.99999999 這種
        # 如果是正的 結果會少1 因此要加回來
        # 如果是負的 結果會多1 因此要減回來
        if float_num > 0 :
            turn_int = int(float_num) + 1
        else : 
            turn_int = int(float_num) - 1
        return turn_int
    else :
        # 會進這裡代表 乘了divide_num之後 還是有小數點以下的數字
        print("error !!!!!!!!!!!!!!!  乘了divide_num之後 還是有小數點以下的數字")
        turn_int = int(float_num)
        # print("ori : " + str_float)
        # print("return : " + str(turn_int))
        return turn_int

# 經過特別處理的比較兩數字
# num1, num2 兩個要比較的東西
# 回傳值:
#   True : 一樣  (包括相差10, 或100倍)
#   False : 不一樣
#   -1 : 超過 100000000 去掉尾數後一樣
# 特殊的方法 : 
#   1. 如果相差10, 或100倍
#   2. 如果超過 100000000 去掉尾數後一樣
def compare_num(num1, num2) :
    # 如果丟進來是"兩個字串" 就直接比完丟回去
    if type(num1) == type(""):
        return num1.strip() == num2.strip()

    # 通用比較
    # 如果是其他資料型態 就直接比較 看一不一樣
    if num1 == num2 : 
        return True
    
    # 前面都不符合 但如果是數字可以用特殊的方法 再比較看看
    elif num1 == num2*10 :
        return True
    elif num1 == num2*100 :
        return True
    elif num1*10 == num2 :
        return True
    elif num1*100 == num2 :
        return True
    
    # 如果數字顯示位數超過8位(到達9位)
    elif num1 >= 100000000 :
        #把最後一位去掉 之後再比一次
        result = compare_num(num1//10, num2)
        if result == True or result == -1 :
            return -1
        else :
            return False
    elif num2 >= 100000000 :
        #把最後一位去掉
        result = compare_num(num1, num2//10)
        if result == True or result == -1 :
            return -1
        else :
            return False
    
    else :
        # print("return False : " + str(num1) +"  :  "+ str(num2))
        return False 

# 比較 list 裡面的項目是否相同 (與list中項目順序無關)
# 通常拿來比手牌前端與後台是否相同
# True : 一樣
# False : 不一樣
# 若 list1 包含 list2 那也會回傳 True
def compare_list(list1, list2, thing = None, cover = False) :
    # 因為怕改到原本的項目 因此複製一份過來比較
    l_big = list1.copy()
    l_small = list2.copy()

    # warning = ""
    
    if thing == None :
        try :
            # 把每一個項目一個一個抓出來 再另一個list中移除此項目
            for things in l_small :
                l_big.remove(things)
        except ValueError:
            # 若跳這個錯誤 代表另一個list沒有此項目
            # 因此不一樣
            return False

    elif thing == "num" :
        try :
            # 把每一個項目一個一個抓出來 再另一個list中移除此項目
            for things in l_small :
                # print(things)
                try :
                    l_big.remove(things)
                except ValueError:
                    if things == 6 :
                        l_big.remove(9)
                    elif things == 9 :
                        l_big.remove(6)
                    else :
                        raise ValueError
                    
        except ValueError:
            # 若跳這個錯誤 代表另一個list沒有此項目
            # 因此不一樣
            return False
    else :
        print("in compare_server.py - compare_list - thing No this type : "+str(thing))
        raise Exception

    # l_big 還有剩東西 或 為0
    if cover == True :
        return True

    if len(l_big) == 0 :
        return True
    else :
        # 另一個list中還有剩餘的東西 代表兩個list不一樣
        return False


    
if __name__ == '__main__' : 
    l1 = [9,6,9]
    l2 = [9,6]
    print(l1)
    print(l2)
    print(compare_list(l1,l2,cover=True))
                

                
