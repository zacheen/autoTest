# import time
# # import datetime
# # import Tool_Main

# color_list = ["黑","紅","梅","方"]
# cal_dict = {}

# for color in color_list :
#     cal_dict[color] = {}
#     for i in range(1,14):
#         cal_dict[color][i] = 0

# cal_dict["鬼"] = 0

# cal_dict["黑"][5] += 1
# cal_dict["梅"][5] += 1
# cal_dict["梅"][5] += 1
# print(cal_dict["黑"][5])


# max_num = 0
# maxx_seat = 0
# color_choose = None
# for color in color_list :
#     for i in range(1,14):
#         max = cal_dict[color][i]
#         if max > max_num:
#             max_num = max
#             max_seat = i
#             color_choose = color 

# print(cal_di。。最多次的牌型為:"+color_choose+str(max_seat))
import pickle
with open('result_count_WXHH.pickle', 'rb') as f:
    read_object = pickle.load(f)
    # print(read_object)

print(read_object["對局牌型"] == read_object["遊戲結果牌型"])
# print(read_object["遊戲結果牌型"])

# color_list = ["黑","紅","梅","方"]
# cal_dict = {}

# for color in color_list :
#     cal_dict[color] = {}
#     for i in range(1,14):
#         cal_dict[color][i] = 0

cal_dict["鬼"] = 0
print(cal_dict)
cal_dict["黑"][1] = 1
print(cal_dict["黑"][1])
# for color in color_list :
#     for i in range(1,14):
#         black_sum = black_sum += cal_dict[color][i]
# print(black_sum)