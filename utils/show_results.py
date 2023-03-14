import os
import cv2


root_txts = '../res_save/PANDA8k/scene1/tiles/bboxes'
root_images = '../data/PANDA8k/scene1/images'
root_output = '../runs/hd/images'


image_names = sorted(os.listdir(root_images))
txt_names = sorted(os.listdir(root_txts))

for frame_index in range(len(txt_names)):
    print(frame_index)
    image = cv2.imread(os.path.join(root_images, str('%s.jpg' % txt_names[frame_index].split('.')[0])))
    f = open(os.path.join(root_txts, txt_names[frame_index]))
    results = f.readlines()
    f.close()
    for res_id in range(len(results)):
        res = results[res_id][:-1].split(' ')
        left = int(res[2])
        top = int(res[3])
        right = int(res[4])
        bottom = int(res[5])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), thickness=3)
    cv2.imwrite(os.path.join(root_output, str('%s.jpg' % txt_names[frame_index].split('.')[0])), image)





