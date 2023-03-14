import os
import json
import cv2
import math


root_images = '../data/PANDA/image_train'
root_annos = '../data/PANDA/image_annos/person_bbox_train.json'
root_output_img = '../runs/hd/images'
root_output_txt = '../runs/hd/annos'

annos = open(root_annos, 'r')
json_annos = json.load(annos)
img_w = 7680
img_h = 4320

keys_list = list(json_annos.keys())

for key_id in range(len(keys_list)):
    key_name = keys_list[key_id]
    seq_name = key_name.split('/')[0]
    if not os.path.exists(os.path.join(root_output_txt, seq_name)):
        os.mkdir(os.path.join(root_output_txt, seq_name))
    obj_list = json_annos[key_name]['objects list']
    f = open(os.path.join(os.path.join(root_output_txt, seq_name), str('%s.txt' % key_name.split('/')[1].split('.')[0])), 'w')
    for obj_id in range(len(obj_list)):
        if obj_list[obj_id]['category'] != 'person':
            continue
        # print(obj_list[obj_id])
        rect_full_body = obj_list[obj_id]['rects']['full body']
        tl = rect_full_body['tl']
        br = rect_full_body['br']
        left = math.ceil(float(tl['x']) * img_w)
        top = math.ceil(float(tl['y']) * img_h)
        right = math.floor(float(br['x']) * img_w)
        bottom = math.floor(float(br['y']) * img_h)
        f.write('person %d %d %d %d\n' % (left, top, right, bottom))
    f.close()

# for key_id in range(len(keys_list)):
#     key_name = keys_list[key_id]
#     print(key_name)
#     seq_name = key_name.split('/')[0]
#     image_path = os.path.join(root_images, key_name)
#     image = cv2.imread(image_path)
#     image_res = cv2.resize(image, (img_w, img_h))
#     if not os.path.exists(os.path.join(root_output_img, seq_name)):
#         os.mkdir(os.path.join(root_output_img, seq_name))
#     cv2.imwrite(os.path.join(os.path.join(root_output_img, seq_name), key_name.split('/')[1]), image_res)



