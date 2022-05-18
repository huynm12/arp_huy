import os

import cv2
import numpy as np
import math
import json
import random

# im_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/7k/images/"
# index_im_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/7k/indexed_images/"
# json_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/7k/annotations/"
# mask_folder = "/home/ubuntu/DEAL/color_label/"

im_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/2k/images/"
index_im_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/2k/indexed_images/"
json_folder = "/vinai-projects/smart-data/ai-modeling/data_huy/2k/annotations/"
mask_folder = "/home/ubuntu/DEAL/test_color_label/"

json_list = os.listdir(json_folder)
print("Processing....")
counter = 0
for json_name in json_list:
    # print("JSON NAMEEEE: ", json_name)
    if json_name[0] == "6":
        counter += 1
        print(counter, "/", len(json_list))
        im_path = os.path.join(im_folder, json_name.replace('json', 'jpeg'))
        indexed_im_path = os.path.join(index_im_folder, json_name.replace('json', 'png'))
        json_path = os.path.join(json_folder, json_name)

        #Indexed image
        index_im = cv2.imread(indexed_im_path)
        index_im = index_im[:, :, 0]

        # color_list = {'Car': [10, 10, 10], ...}
        color_list = {}
        with open("fixed_color.txt", 'r') as f:
            for line in f.readlines():
                line_split = line.split(';')
                key = line_split[0]
                color = line_split[1]
                # key, color = ' '.split(line)[0], ' '.split(line)[1]
                color_split = color.split(',')
                color_int = [int(x) for x in color_split]
                color_list[key] = color_int


        #Json
        json_file = open(json_path, 'r')
        data = json.load(json_file)

        label_data = data['response']['labelMapping']
        # print(label_data)
        label_list = []
        for key in label_data.keys():
            print(key)
            label_list.append(key)

        mask = np.zeros(list(index_im.shape)+[3])
        # print("LABEL_DATA: ", label_data)
        # i=0
        for key, value in label_data.items():
            # i+=1
            # print("Index: ",i," VALUEEEEEEEE: ",value)
            if type(value) is list:
                if len(value) > 0:
                    color_id = value[0]['index']
                    for i in range(len(value)):
                        id = value[i]['index']
                        mask[index_im == id] = color_list[key]
            else:
                if len(value) > 0:
                    id = value['index']
                    mask[index_im == id] = color_list[key]

        mask = np.uint8(mask)
        # mask = cv2.resize(mask, (640,480))
        # im = cv2.resize(im, (640,480))
        mask_name = json_name.replace('json', 'png')
        cv2.imwrite(os.path.join(mask_folder, mask_name), mask)
    else:
        continue

    # cv2.imshow("hjhj mask ne", mask)
    # cv2.imshow("hjhj im ne", im)
    # cv2.waitKey()
    # for i in range(data)