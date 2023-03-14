import os
import cv2
import json


root_images = './data/PANDA/image_train/02_Xili_Crossroad'
root_annos = './data/PANDA/image_annos/person_bbox_train.json'


annos = open(root_annos, 'r')
json_annos = json.load(annos)


keys_list = list(json_annos.keys())
print(keys_list)
for i in range(len(keys_list)):
    print(keys_list[i])
print(len(json_annos['01_University_Canteen/IMG_01_01.jpg']['objects list']))
print(json_annos['01_University_Canteen/IMG_01_01.jpg']['objects list'][0])


