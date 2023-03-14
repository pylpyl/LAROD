import os
import cv2


root_images = '../data/PANDA8k/detection/images'
root_annos = '../data/PANDA8k/detection/annos'
root_output = '../runs/hd/images'


seqs = sorted(os.listdir(root_images))


for seq_id in range(len(seqs)):
    seq_name = seqs[seq_id]
    print(seq_name)
    image_names = sorted(os.listdir(os.path.join(root_images, seq_name)))
    for image_id in range(len(image_names)):
        image_name = image_names[image_id]
        image_res = cv2.imread(os.path.join(os.path.join(root_images, seq_name), image_name))
        txt_name = str('%s.txt' % image_name.split('.')[0])
        print(image_name)
        f = open(os.path.join(os.path.join(root_annos, seq_name), txt_name))
        lines = f.readlines()
        f.close()
        for line in lines:
            info = line[:-1].split(' ')
            left = int(info[1])
            top = int(info[2])
            right = int(info[3])
            bottom = int(info[4])
            cv2.rectangle(image_res, (left, top), (right, bottom), (0, 0, 255), thickness=3)
        if not os.path.exists(os.path.join(root_output, seq_name)):
            os.mkdir(os.path.join(root_output, seq_name))
        cv2.imwrite(os.path.join(os.path.join(root_output, seq_name), image_name), image_res)




# image_names = sorted(os.listdir(root_images))
# anno_names = sorted(os.listdir(root_annos))
#
# for frame_index in range(len(image_names)):
#     print(frame_index)
#     image = cv2.imread(os.path.join(root_images, image_names[frame_index]))
#     f = open(os.path.join(root_annos, anno_names[frame_index]))
#     annos = f.readlines()
#     f.close()
#     for anno_id in range(len(annos)):
#         anno = annos[anno_id][:-1].split(' ')
#         left = int(anno[1])
#         top = int(anno[2])
#         right = int(anno[3])
#         bottom = int(anno[4])
#         cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), thickness=3)
#     cv2.imwrite(os.path.join(root_output, image_names[frame_index]), image)





