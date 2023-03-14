import os
import cv2


root_images = '../data/PANDA8k/round2_tracking/scene3/images'
root_annos = '../data/PANDA8k/round2_tracking/scene3/annos'
root_output = '../runs/hd/images'


image_names = sorted(os.listdir(root_images))
anno_names = sorted(os.listdir(root_annos))

for frame_index in range(len(image_names)):
    print(frame_index)
    image = cv2.imread(os.path.join(root_images, image_names[frame_index]))
    f = open(os.path.join(root_annos, anno_names[frame_index]))
    annos = f.readlines()
    f.close()
    for anno_id in range(len(annos)):
        anno = annos[anno_id][:-1].split(' ')
        left = int(anno[1])
        top = int(anno[2])
        right = int(anno[3])
        bottom = int(anno[4])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), thickness=3)
    cv2.imwrite(os.path.join(root_output, image_names[frame_index]), image)





