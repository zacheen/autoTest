# 在辨識以前通常都是用這個順序 ["鬼","黑","紅","梅","方"]

# 大王 : Card(2, "鬼") 小王 : Card(1, "鬼")

class Card():
    # 目前是用preset_suit_order來比較花色大小
    preset_suit_order = ["鬼","黑","紅","梅","方"]
    # init 的時候可以來這邊 取值覆寫掉suit_order
    # 由此決定各個遊戲的花色大小順序
    other_game_suit_order = {}


    # point 可以代入字串(EX: "A", "Q", "10")或數字 都會自動轉成 int
    # suit 可以代入字串(EX: "黑桃", "愛心")或後台格式字串 (EX:"♠", "♥")
    def __init__(self,point,suit) : 
        # 這個 class 有兩個參數
        # 1. point 代表點數
        # 2. suit 代表花色

        # 處理文字成point
        if point == "A" :
            self.point = 1
        elif point == "J" :
            self.point = 11
        elif point == "Q" :
            self.point = 12
        elif point == "K" :
            self.point = 13
        else :
            # 其他就直接轉 int 
            self.point = int(point)
            
        # 處理文字成suit
        if suit == "♠" or suit == "黑桃" or suit == "黑":
            self.suit = "黑"
        elif suit == "♥" or suit == "愛心" or suit == "紅" :
            self.suit = "紅"
        elif suit == "♦" or suit == "菱形" or suit == "方塊" or suit == "方" :
            self.suit = "方"
        elif suit == "♣" or suit == "梅花" or suit == "梅" :
            self.suit = "梅"
        elif suit == "王" or suit == "鬼牌" or suit == "鬼" :
            self.suit = "鬼"
        else:
            print("Error 沒有此花色!!")
            raise Exception

    def __str__(self) :
        # 呼叫 print() 的時候會自動呼叫這一個 function
        # 此格式的輸出複製貼上可以初始化成一個 Card 的 Class
        return " Card("+str(self.point) + ", \"" + str(self.suit)+"\")"
        # return "point: " + str(self.point) + " , suit: " + str(self.suit)

    def __repr__(self) :
        # print List 會使用此function來輸出
        return str(self)

    def __gt__(self, other) :
        # 先比點數大小 再比花色
        if self.point == other.point : 
            # 如果點數一樣才會進來
            # 取得 這個花色 的 順序位置
            s1 = self.preset_suit_order.index(self.suit)
            s2 = self.preset_suit_order.index(other.suit)
            # 比較花色順序大小
            if s1 == s2 :
                # !!!!!! 此處尚未定義 兩張牌一樣要回傳什麼 !!!!!!!!!
                # print("靠北阿 有一樣的牌!!!!!!!")
                return True
            else :
                # 注意 s1 愈小牌愈大 (因為位置愈前面愈大)
                return s1 < s2
        else : 
            # 比較point點數順序大小 
            # 注意 point 愈大 牌愈大
            return self.point > other.point

    def __eq__(self,other):   # 定義 == 運算
        # if self.point == other.point and self.suit == other.suit :
        #     return True
        # else :
        #     return False
        return self.__dict__ == other.__dict__

# 修改花色大小順序用 如果 suit_order = None 代表使用預設的花色大小 ['鬼', '黑', '紅', '梅', '方']
# 使用方法 : 
#       change_suit_order(['鬼', '黑', '紅', '方', '梅'])
#       通常只有這兩種花色大小
def change_suit_order(suit_order = None):
    if suit_order != None :
        Card.preset_suit_order = suit_order
    print("目前使用的花色大小順序(大->小) : "+str(Card.preset_suit_order))

# 把字串轉成 Card 的 list 回傳
# 輸入字串格式 EX: 
# 大王♠10♦5
# 大王 ♠10 ♦5
# 如果回傳為空的list 代表此字串不是牌
def chagne_str_to_list(card_str) : 
    # 初始化要回傳的list
    card_list = []

    # 先把 大王 小王 挑出來 加進list中
    if "大王" in card_str :
        card_str = card_str.replace("大王","")
        card_list.append(Card(2, "鬼"))
    if "小王" in card_str :
        card_str = card_str.replace("小王","")
        card_list.append(Card(1, "鬼"))
    # print(card_str)

    # str_max_index 紀錄str_pos_count最大可以到多少 超過要跳出迴圈
    str_max_index = len(card_str)-1
    # str_pos_count 紀錄現在處理到card_str第幾個字
    str_pos_count = 0
    while True : 
        # str_pos_count最大可以到多少 超過要跳出迴圈
        if str_pos_count >= str_max_index :
            break
        else :
            # 如果這個字是空白 就跳過這個字
            if card_str[str_pos_count] == " ":
                str_pos_count+=1
        
        # 處理這個文字成suit
        if card_str[str_pos_count] == "♠" :
            suit = '黑'
        elif card_str[str_pos_count] == "♥" :
            suit = '紅'
        elif card_str[str_pos_count] == "♦" :
            suit = '方'
        elif card_str[str_pos_count] == "♣" :
            suit = '梅'
        else:
            break
        # 處理這個文字完 +1 換下一個位置
        str_pos_count+=1

        # 初始化
        point = 0
        # print(card_str[str_pos_count])
        # print("len", len(card_str))
        # 處理這個文字成point
        # 因為不知道數字是幾個位數　因此要用while迴圈偵測
        # while偵測有沒有到字串上限
        while str_pos_count < len(card_str):
            # 把 "10" 轉成 int 10 的好用 code
            if card_str[str_pos_count] >= "0" and card_str[str_pos_count] <= "9" :
                point = point*10 + int(card_str[str_pos_count])
            elif card_str[str_pos_count] == "A" :
                point = 1
            elif card_str[str_pos_count] == "J" :
                point = 11  
            elif card_str[str_pos_count] == "Q" :
                point = 12
            elif card_str[str_pos_count] == "K" :
                point = 13
            else :
                # 如果不是數字或"A","J".... 就跳出
                break
            str_pos_count+=1
        
        # 把結果加進list
        card_list.append(Card(point, suit))

    return card_list

# 尚未實作
# EX : 
# chagne_link_to_list("tao7")
# chagne_link_to_list("hong6")
# chagne_link_to_list("caoc")
check_list = ["1","2","3","4","5","6","7","8","9","a","b","c","d"]
def chagne_link_to_list(link_str) : 
    if link_str[0] == "t":
        color = "黑"
        num = check_list.index(link_str[3])+1
    elif link_str[0] == "h":
        color = "紅"
        num = check_list.index(link_str[4])+1
    elif link_str[0] == "c":
        color = "梅"
        num = check_list.index(link_str[3])+1
    elif link_str[0] == "f":
        color = "方"
        num = check_list.index(link_str[4])+1
    else:
        print("遊戲結果無法判斷花色")

    return Card(num,color)

if __name__ == '__main__' :
    pass
    # 創建方法1
    # print(chagne_str_to_list("大王♥5♦K♥3"))
    # print(chagne_str_to_list("大王 小王"))
    print(chagne_str_to_list("西八"))
    print(chagne_str_to_list("♥5shark"))
    print(chagne_str_to_list("shark♥5"))
    print(chagne_str_to_list("shark大王"))

    # 創建方法2
    # c1 = Card(11, "梅")
    # print(c1)
    # c1 = Card(12, "♥")
    # print(c1)
    # c2 = Card(12, "方")
    # print(c2)
    # print(c1 > c2)
    # ll = [Card(12, '黑'), Card(12, '方'), Card(12,'紅'), Card(1,'黑'), Card(2, '黑')]
    # ll.sort(reverse=True)
    # print(ll)

    # change_suit_order(['鬼', '紅', '方', '梅', '黑'])
    # ll = [Card(12, '黑'), Card(12, '方'), Card(12,'紅'), Card(1,'黑'), Card(2, '黑')]
    # ll.sort(reverse=True)
    # print(ll)

    # print(chagne_link_to_list("caoc"))
    # print(chagne_link_to_list("tao6"))