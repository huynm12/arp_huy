import os
import shutil

train_data_path = "/vinai-projects/smart-data/ai-modeling/data_huy/7k/images/"
test_data_path = "/vinai-projects/smart-data/ai-modeling/data_huy/2k/images/"

label_path = "/home/ubuntu/DEAL/color_label"
test_label_path = "/home/ubuntu/DEAL/test_color_label"

os.mkdir(os.path.join("/home/ubuntu/DEAL", "autopilot_scale"))
os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "train"))
os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "val"))
os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "test"))

os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "train_labels"))
os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "val_labels"))
os.mkdir(os.path.join("/home/ubuntu/DEAL/autopilot_scale", "test_labels"))

train_image_list = os.listdir(train_data_path)
test_image_list = os.listdir(test_data_path)

total_train_image = len(train_image_list)
count = 0
for image in train_image_list:
    if 'jpeg' in image:
        count+=1
        print("----------COUNT---------: ", count)
        if count <= int(total_train_image * 0.9):
            shutil.copy(os.path.join(train_data_path,image), os.path.join("/home/ubuntu/DEAL/autopilot_scale", "train"))
            shutil.copy(os.path.join(label_path,image.replace("jpeg", "png")), os.path.join("/home/ubuntu/DEAL/autopilot_scale", "train_labels"))
        else:
            shutil.copy(os.path.join(train_data_path,image), os.path.join("/home/ubuntu/DEAL/autopilot_scale", "val"))
            shutil.copy(os.path.join(label_path,image.replace("jpeg", "png")), os.path.join("/home/ubuntu/DEAL/autopilot_scale", "val_labels"))

for image in test_image_list:
    if 'jpeg' in image:
        shutil.copy(os.path.join(test_data_path,image),os.path.join("/home/ubuntu/DEAL/autopilot_scale", "test"))
        shutil.copy(os.path.join(test_label_path,image.replace("jpeg", "png")), os.path.join("/home/ubuntu/DEAL/autopilot_scale", "test_labels"))




