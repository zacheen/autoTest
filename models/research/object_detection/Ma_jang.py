# 萬 1-9 筒 21-29 條 41-49 
word_list = [ "東", "南", "西", "北", "中", "發", "白",    "春", "夏", "秋", "冬", "梅", "兰", "竹", "菊"]
#              60   70    80    90   100   110   120      1000  2000  3000 4000  5000  6000  7000  8000 


from Gf_Except import Game_fail_Exception

# 因為麻將的牌不會連起來 
def chagne_str_to_list(card_str):
    # print("card_str", card_str)
    
    card_list = []

    # print(card_str)
    str_max_index = len(card_str)-1
    str_pos_count = 0
    while True : 
        if str_pos_count > str_max_index :
            # print(str_pos_count,str_max_index)
            break

        # print(str_pos_count,":",card_str[str_pos_count])

        if card_str[str_pos_count] == " " :
            str_pos_count+=1
            continue
        
        if card_str[str_pos_count] == "东" and card_str[str_pos_count+1] == "风" :
            str_pos_count+=2
            card_list.append(60)
            continue
        elif card_str[str_pos_count] == "南" and card_str[str_pos_count+1] == "风" :
            str_pos_count+=2
            card_list.append(70)
            continue
        elif card_str[str_pos_count] == "西" and card_str[str_pos_count+1] == "风" :
            str_pos_count+=2
            card_list.append(80)
            continue
        elif card_str[str_pos_count] == "北" and card_str[str_pos_count+1] == "风"  :
            str_pos_count+=2
            card_list.append(90)
            continue
        elif card_str[str_pos_count] == "白" :
            str_pos_count+=1
            card_list.append(120)
            continue
        elif card_str[str_pos_count] == "發" :
            str_pos_count+=1
            card_list.append(110)
            continue
        elif card_str[str_pos_count] == "中":
            str_pos_count+=1
            card_list.append(100)
            continue
        elif card_str[str_pos_count] == "紅" and card_str[str_pos_count+1] == "中":
            str_pos_count+=2
            card_list.append(100)
            continue
        elif card_str[str_pos_count] == "春":
            str_pos_count+=1
            card_list.append(1000)
            continue
        elif card_str[str_pos_count] == "夏":
            str_pos_count+=1
            card_list.append(2000)
            continue
        elif card_str[str_pos_count] == "秋":
            str_pos_count+=1
            card_list.append(3000)
            continue
        elif card_str[str_pos_count] == "冬":
            str_pos_count+=1
            card_list.append(4000)
            continue
        elif card_str[str_pos_count] == "梅":
            str_pos_count+=1
            card_list.append(5000)
            continue
        elif card_str[str_pos_count] == "兰":
            str_pos_count+=1
            card_list.append(6000)
            continue
        elif card_str[str_pos_count] == "竹":
            str_pos_count+=1
            card_list.append(7000)
            continue
        elif card_str[str_pos_count] == "菊":
            str_pos_count+=1
            card_list.append(8000)
            continue


        if card_str[str_pos_count] == "一" :
            num = 1
        elif card_str[str_pos_count] == "二" :
            num = 2
        elif card_str[str_pos_count] == "三" :
            num = 3
        elif card_str[str_pos_count] == "四" :
            num = 4
        elif card_str[str_pos_count] == "五" :
            num = 5
        elif card_str[str_pos_count] == "六" :
            num = 6
        elif card_str[str_pos_count] == "七" :
            num = 7
        elif card_str[str_pos_count] == "八" :
            num = 8
        elif card_str[str_pos_count] == "九" :
            num = 9
        else:
            break


        str_pos_count+=1

        if card_str[str_pos_count] == "万" :
            card_list.append(num)
        elif card_str[str_pos_count] == "筒" :
            card_list.append(20+num)
        elif card_str[str_pos_count] == "条" :
            card_list.append(40+num)
        else:
            break
        str_pos_count+=1

    return card_list

# chagne_link_to_list("31") # 一餅
# chagne_link_to_list("21") # 一條
# chagne_link_to_list("15") # 5萬
def chagne_link_to_list(link_str) : 
    
    link_num = int(link_str)

    if link_num//10 == 1 :
        return link_num%10
    elif link_num//10 == 3 :
        return 20+link_num%10
    elif link_num//10 == 2 :
        return 40+link_num%10

def find_red(hand_card) : 
    red_count = 0
    while 100 in hand_card :
        hand_card.remove(100)
        red_count = red_count+1
    return hand_card, red_count

def pass_list( final_com, new_item ) : 
    pass_final_com = final_com.copy()
    pass_final_com.append(new_item)
    # print(pass_final_com)
    return pass_final_com

def print_card(all_listen_set, pri = False) :
    if "all" in all_listen_set : 
        lis_list = "全部皆可胡"
    
    else :
        lis_list = []
        for x in all_listen_set :
            if x >= 1 and x <= 9 :
                lis_list.append(str(x)+"萬")
            
            elif x >= 21 and x <= 29 :
                lis_list.append(str(x-20)+"筒")

            elif x >= 41 and x <= 49 :
                lis_list.append(str(x-40)+"條")
            
            elif x > 60 :
                pass_num = x//10 
                lis_list.append(word_list[pass_num-6])

    if pri :
        print(lis_list)
    
    return lis_list

def judge_hoo(hand_card, red_count, eye_count, gong_count, final_com) :
    list_len = len(hand_card)
    total_card = len(hand_card) + red_count
    
    # print(str(final_com) + "   red_count : " + str(red_count) + "   eye_count : " + str(eye_count)+ "   gong_count : " + str(gong_count))
    if len(hand_card) == 0 :
        # 把紅中加回
        if red_count > 0 :
            red_list = []
            for x in range(red_count) :
                red_list.append(100)
            final_com.append(red_list)
        
        if eye_count == 1 and gong_count == 0 : 
            # print("sol : "+str(final_com))
            return True
        else :
            print("something wrong!!!!!!!!")
            return False

    if list_len >= 2 :
        # 有兩張一樣
        if hand_card[0] == hand_card[1] : 
            if list_len >= 3 :
                if hand_card[0] == hand_card[2] :
                    if list_len >= 4 :
                        # 槓 
                        if gong_count >= 1 : 
                            if hand_card[0] == hand_card[3] : 
                                if judge_hoo(hand_card[4:], red_count, eye_count, gong_count-1, pass_list(final_com, hand_card[0:4])) : 
                                    return True
                    
                    # 刻子
                    if judge_hoo(hand_card[3:], red_count, eye_count, gong_count, pass_list(final_com, hand_card[0:3])) :            
                        return True
            
            if list_len >= 2 :
                # 中+刻子
                if red_count >= 1 :
                    # 傳遞參數用
                    new_hand_card = hand_card[0:2]
                    new_hand_card.append(100)
                    if judge_hoo(hand_card[2:], red_count-1, eye_count, gong_count, pass_list(final_com, new_hand_card)) : 
                        return True
                
                # 眼
                if eye_count < 1 : 
                    if judge_hoo(hand_card[2:], red_count, eye_count+1, gong_count, pass_list(final_com, hand_card[0:2])) : 
                        return True

        # 有兩張相差1
        if (hand_card[0]+1) in hand_card : 
            if list_len >= 3 :
                # 順子
                if (hand_card[0]+2) in hand_card : 
                    # 傳遞參數用
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+1)
                    new_hand_card.remove(hand_card[0]+2)
                    if judge_hoo(new_hand_card, red_count, eye_count, gong_count, pass_list(final_com, [hand_card[0],hand_card[0]+1,hand_card[0]+2])) : 
                        return True

            if list_len >= 2 :
                # 中+順子(前後)
                if red_count >= 1 : 
                    # 傳遞參數用
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+1)
                    if judge_hoo(new_hand_card, red_count-1, eye_count, gong_count, pass_list(final_com, [hand_card[0],hand_card[0]+1,100])) : 
                        return True
                
        # 中+順子(中洞)
        if list_len >= 2 :
            if (hand_card[0]+2) in hand_card : 
                if red_count >= 1 :
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+2)
                    if judge_hoo(new_hand_card, red_count-1, eye_count, gong_count, pass_list(final_com, [hand_card[0],100,hand_card[0]+2])) : 
                        return True

    # 中+眼 
    if eye_count < 1 and red_count >= 1: 
        if judge_hoo(hand_card[1:], red_count-1, eye_count+1, gong_count, pass_list(final_com, [hand_card[0],100])) : 
            return True

    # 2中+一張牌(可以是順，可以是刻)
    if red_count >= 2 : 
        if judge_hoo(hand_card[1:], red_count-2, eye_count, gong_count, pass_list(final_com, [hand_card[0],100,100])) : 
            return True


    if total_card >= 14 :
        # 4紅中
        if red_count == 4 :
            final_com.append([100,100,100,100])
            # print("sol : "+str(final_com))
            # print("4紅中")
            return True
        
        # 7對
        seven_pair = []
        while True :
            if hand_card[0] == hand_card[0+1] :
                seven_pair.append([hand_card[0],hand_card[1]])
                hand_card = hand_card[2:]
            # 代表這張是單獨的
            elif red_count >= 1 :
                red_count = red_count-1
                seven_pair.append([hand_card[0],100])
                hand_card = hand_card[1:]
            else :
                return False

            # 剩一張
            if len(hand_card) == 1 :
                if red_count >= 1 :
                    red_count = red_count-1
                    seven_pair.append([hand_card[0],100])
                    hand_card = hand_card[1:]

            # 最後一對結束
            if len(hand_card) == 0 :
                if red_count > 0 :
                    seven_pair.append([100,100])
                final_com.append(seven_pair)
                print("sol : "+str(final_com))
                return True
      
    return False

def do_judge_hoo(hand_card) : 
    # print("現在手牌有 "+str(len(hand_card))+" 張   (通常是14張)")
    if len(hand_card) > 16 :
        gong_count = len(hand_card)-14
    else :
        gong_count = (len(hand_card)-2) % 3
    hand_card.sort()
    # print(hand_card)

    after_find_red, red_count = find_red(hand_card)
    # print(judge_hoo(after_find_red, red_count, 0, gong_count, []))
    return judge_hoo(after_find_red, red_count, 0, gong_count, [])


def pass_set(all_listen, new_items) : 
    all_listen = all_listen.copy()

    if type(new_items) == type([]):
        for item in new_items :
            all_listen.add(item)
    else :
        all_listen.add(new_items)

    return all_listen

all_combination = []
all_listen_set = {0}
def listen(hand_card, red_count, eye_count, gong_count, final_com, all_listen) :
    global all_combination
    global all_listen_set

    total_card = len(hand_card) + red_count
    
    # print(str(final_com) + "   red_count : " + str(red_count) + "   eye_count : " + str(eye_count)+ "   gong_count : " + str(gong_count))

    if len(hand_card) == 0 :
        # 2中 1中
        if red_count > 0 :
            all_listen.add("all")
        else :
            pass

        if eye_count == 1 and gong_count == 0 :
            print("sol : "+str(final_com.copy()))
            all_combination.append(final_com.copy())
            all_listen_set = all_listen_set.union(all_listen)
        else :
            print("something wrong!!!!!!!!")
            
        return

    list_len = len(hand_card)
    if list_len >= 2 :
        # 有兩張一樣
        if hand_card[0] == hand_card[1] : 
            if list_len >= 3 :
                if hand_card[0] == hand_card[2] :
                    if list_len >= 4 :
                        # 槓 
                        if gong_count >= 1 : 
                            if hand_card[0] == hand_card[3] : 
                                if listen(hand_card[4:], red_count, eye_count, gong_count-1, pass_list(final_com, hand_card[0:4]), all_listen.copy()) : 
                                    pass
                    
                    # 刻子
                    if listen(hand_card[3:], red_count, eye_count, gong_count, pass_list(final_com, hand_card[0:3]), all_listen.copy()) :            
                        pass
            
            if list_len >= 2 :
                # 中+刻子
                if red_count >= 1 :
                    # 傳遞參數用
                    new_hand_card = hand_card[0:2]
                    new_hand_card.append(100)
                    if listen(hand_card[2:], red_count-1, eye_count, gong_count, pass_list(final_com, new_hand_card), pass_set(all_listen, hand_card[0])) : 
                        pass
                
                # 眼
                if eye_count < 1 : 
                    if listen(hand_card[2:], red_count, eye_count+1, gong_count, pass_list(final_com, hand_card[0:2]), all_listen.copy()) : 
                        pass

        # 有兩張相差1
        if (hand_card[0]+1) in hand_card : 
            if list_len >= 3 :
                # 順子
                if (hand_card[0]+2) in hand_card : 
                    # 傳遞參數用
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+1)
                    new_hand_card.remove(hand_card[0]+2)
                    if listen(new_hand_card, red_count, eye_count, gong_count, pass_list(final_com, [hand_card[0],hand_card[0]+1,hand_card[0]+2]), all_listen.copy()) : 
                        pass

            if list_len >= 2 :
                # 中+順子(前後)
                if red_count >= 1 : 
                    # 傳遞參數用
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+1)
                    if listen(new_hand_card, red_count-1, eye_count, gong_count, pass_list(final_com, [hand_card[0],hand_card[0]+1,100]), pass_set(all_listen, [hand_card[0]-1, hand_card[0]+2])) : 
                        pass
                
        # 中+順子(中洞)
        if list_len >= 2 :
            if (hand_card[0]+2) in hand_card : 
                if red_count >= 1 :
                    new_hand_card = hand_card[1:]
                    new_hand_card.remove(hand_card[0]+2)
                    if listen(new_hand_card, red_count-1, eye_count, gong_count, pass_list(final_com, [hand_card[0],100,hand_card[0]+2]), pass_set(all_listen,hand_card[0]+1)) : 
                        pass

    # 中+眼 
    if eye_count < 1 and red_count >= 1: 
        if listen(hand_card[1:], red_count-1, eye_count+1, gong_count, pass_list(final_com, [hand_card[0],100]), pass_set(all_listen, hand_card[0])) : 
            pass

    # 2中+一張牌(可以是順，可以是刻)
    if red_count >= 2 : 
        if listen(hand_card[1:], red_count-2, eye_count, gong_count, pass_list(final_com, [hand_card[0],100,100]), pass_set(all_listen, [hand_card[0]-2,hand_card[0]-1,hand_card[0],hand_card[0]+1,hand_card[0]+2])) : 
            pass
    
    if total_card >= 14 :
        # 4紅中
        if red_count == 5 :
            final_com.append([100,100,100,100])
            print("sol : "+str(final_com))
            all_combination.append(final_com.copy())
            all_listen.add("all")
            all_listen_set = all_listen_set.union(all_listen)
            print("4紅中")
            # 全聽

        # 3紅中
        if red_count == 4 :
            final_com.append([100,100,100,100])
            print("sol : "+str(final_com))
            all_combination.append(final_com.copy())
            all_listen.add("100")
            all_listen_set = all_listen_set.union(all_listen)
            print("3紅中")
            # 聽紅中
        
        # 7對
        seven_pair = []
        while True :
            if hand_card[0] == hand_card[0+1] :
                seven_pair.append([hand_card[0],hand_card[1]])
                hand_card = hand_card[2:]
            # 代表這張是單獨的
            elif red_count >= 1 :
                red_count = red_count-1
                all_listen.add(hand_card[0])
                seven_pair.append([hand_card[0],100])
                hand_card = hand_card[1:]
            else :
                break

            # 剩一張
            if len(hand_card) == 1 :
                if red_count >= 1 :
                    red_count = red_count-1
                    all_listen.add(hand_card[0])
                    seven_pair.append([hand_card[0],100])
                    hand_card = hand_card[1:]

            # 最後一對結束
            if len(hand_card) == 0 :
                if red_count > 0 :
                    all_listen.add("all")
                    seven_pair.append([100,100])
                all_combination.append(seven_pair)
                all_listen_set = all_listen_set.union(all_listen)
                break
        return all_listen

def do_listen(hand_card) : 
    global all_combination
    global all_listen_set
    all_combination = []
    all_listen_set = {0}

    print("現在手牌有 "+str(len(hand_card))+" 張   (通常是13張)")
    if len(hand_card) > 16 :
        gong_count = len(hand_card)-13
    else :
        gong_count = (len(hand_card)-1) % 3
    hand_card.sort()
    print_card(hand_card, True)

    after_find_red, red_count = find_red(hand_card)
    print_card(hand_card, True)
    
    # red_count要加1變成聽的那一張 
    listen(after_find_red, red_count+1, 0, gong_count, [], {0})

    print("all_combination")
    for com in all_combination : 
        print(com)
    print_card(all_listen_set, True)


def count_four_card(card_list):
    card_list.sort()
    return_list = []

    remember_card = None
    count_num = 0
    for this_card in card_list :
        if remember_card == this_card :
            count_num += 1
            if count_num == 4 :
                return_list.append(remember_card)
        else:
            remember_card = this_card
            count_num = 1

    return return_list


if __name__=="__main__" : 
    print(chagne_link_to_list("31")) # 一餅
    print(chagne_link_to_list("21")) # 一條
    print(chagne_link_to_list("15")) # 5萬
    #--------------------------------------------------------------------------------------
    # hand_card = [2,3,4,22,22,22,23,24,25,26]
    # # if len(hand_card) < 14 :
    # #     print("靠北喔 現在只有 "+str(len(hand_card))+" 張  不到14張啦")
    # do_listen(hand_card)
    # print("-----------------------------------------------------------------------------------------------")
    # hand_card = [1,2,3,5,5,5,6,6,7,8,4]
    # do_judge_hoo(hand_card)
    # print("-----------------------------------------------------------------------------------------------")
    # print(chagne_str_to_list("七筒 四条 五条 一万"))

    # print(count_four_card([1, 41, 41, 41, 41, 42, 43, 45, 45, 45, 45, 46, 47, 49, 49, 42]))