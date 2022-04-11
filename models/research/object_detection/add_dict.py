
# [第一次行動][聽牌腳本][是否被加注]
# [第二次行動][聽牌腳本][牌型][是否被加注]
# [0-1][0-5][]
# def add_item_result_count(result_count, index_list, value) :
#     if len(index_list) == 1 :
#         result_count[index_list[0]] = value
#         return

#     elif index_list[0] not in result_count.keys() :
#         result_count[index_list[0]] = {}

#         add_item_result_count(result_count[index_list[0]], index_list[1:], value)

#     else :
#         add_item_result_count(result_count[index_list[0]], index_list[1:], value)

def add_item_count(result_count, index_list) :
    if len(index_list) == 1 :
        if index_list[0] not in result_count.keys() :
            # 代表第一次進還沒有 0
            result_count[index_list[0]] = 1
        else :
            result_count[index_list[0]] += 1
        
        return

    elif index_list[0] not in result_count.keys() :
        result_count[index_list[0]] = {}

        add_item_count(result_count[index_list[0]], index_list[1:])

    else :
        add_item_count(result_count[index_list[0]], index_list[1:])

if __name__ == "__main__":
    result_count = {}
    add_item_count(result_count, ["level 1","level 2","a"])
    add_item_count(result_count, ["level 1","level 2","b"])
    add_item_count(result_count, ["level 1","c"])
    add_item_count(result_count, ["level 1","level 2","a"])
    print(result_count)