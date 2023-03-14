import os
import cv2


root_input_images = '../data/PANDA8k/round2_tracking/scene10/images'
root_input_annos = '../data/PANDA8k/round2_tracking/scene10/annos_original'
root_output_annos = '../runs/hd/annos'
small_obj_threshold = 50


annos = sorted(os.listdir(root_input_annos))


for anno_id in range(len(annos)):
# for anno_id in range(1):
    anno_name = annos[anno_id]
    f = open(os.path.join(root_input_annos, anno_name))
    anno_items = f.readlines()
    f.close()
    anno_items_res = []
    for anno_item_id in range(len(anno_items)):
        anno_item = anno_items[anno_item_id][:-1]
        anno = anno_item.split(' ')
        left = int(anno[1])
        top = int(anno[2])
        right = int(anno[3])
        bottom = int(anno[4])
        if right - left < small_obj_threshold and bottom - top < small_obj_threshold:
            continue
        if anno_item not in anno_items_res:
            anno_items_res.append(anno_item)
    f = open(os.path.join(root_output_annos, anno_name), 'w')
    for anno_item_id in range(len(anno_items_res)):
        f.write('%s\n' % anno_items_res[anno_item_id])
    f.close()



