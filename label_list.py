import json
import numpy as np
import cv2
import os

label_list = []
train_annotations_folder_path = "/vinai-projects/smart-data/ai-modeling/data_huy/7k/annotations/"
for json_name in os.listdir(train_annotations_folder_path):
    if json_name[0] == "6":
        print("json_name_train = ", json_name)
        json_path = os.path.join(train_annotations_folder_path, json_name)
        json_file = open(json_path, 'r')
        data = json.load(json_file)

        label_data = data['response']['labelMapping']
        for key in label_data.keys():
            if key not in label_list:
                label_list.append(key)
    else:
        continue

test_annotations_folder_path = "/vinai-projects/smart-data/ai-modeling/data_huy/2k/annotations/"
for json_name in os.listdir(test_annotations_folder_path):
    if json_name[0] == "6":
        print("json_name_test = ", json_name)
        json_path = os.path.join(test_annotations_folder_path, json_name)
        json_file = open(json_path, 'r')
        data = json.load(json_file)

        label_data = data['response']['labelMapping']
        for key in label_data.keys():
            if key not in label_list:
                label_list.append(key)
    else:
        continue

# a.write("Car 10 10 10\n")
random_check_list = []
color_file = open("fixed_color.txt", 'w+')
for i in range(len(label_list)):
    print("i ======== ", i)
    random_color = np.random.choice(range(256), size=3)
    color_str = str(random_color)
    while color_str in random_check_list:
        random_color = np.random.choice(range(256), size=3)
    random_check_list.append(color_str)

    random_color_list = list(random_color)
    random_color_string = ','.join(map(str, random_color_list))
    color_file.write(label_list[i] + ";" + random_color_string + "\n")
color_file.close()

# color_file = open("fixed_color.txt", 'r')
# print(color_file.readlines())
