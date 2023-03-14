import os
import json
import cv2


root_images = '../data/PANDA/round2_tracking/train_part10/10_Huaqiangbei'
root_annos = '../data/PANDA/round2_tracking/train_annos/10_Huaqiangbei/tracks.json'
root_output_img = '../runs/hd/images'
root_output_txt = '../runs/hd/annos'

annos = open(root_annos, 'r')
json_annos = json.load(annos)
img_w = 7680
img_h = 4320

frame_names = sorted(os.listdir(root_images))
for image_id in range(len(frame_names)):
    image = cv2.imread(os.path.join(root_images, frame_names[image_id]))
    image_res = cv2.resize(image, (img_w, img_h))
    anno_txt_name = str('%s.txt' % frame_names[image_id].split('.')[0])
    frame_id = image_id + 1
    print('frame id %d' % frame_id)
    f = open(os.path.join(root_output_txt, anno_txt_name), 'w')
    for anno_index in range(len(json_annos)):
        person_track_id = int(json_annos[anno_index]['track id'])
        for person_frame_index in range(len(json_annos[anno_index]['frames'])):
            person_frame_id = int(json_annos[anno_index]['frames'][person_frame_index]['frame id'])
            if person_frame_id == frame_id:
                if not json_annos[anno_index]['frames'][person_frame_index]['occlusion'] in ['normal']:
                    continue
                left = round(float(json_annos[anno_index]['frames'][person_frame_index]['rect']['tl']['x']) * img_w)
                top = round(float(json_annos[anno_index]['frames'][person_frame_index]['rect']['tl']['y']) * img_h)
                right = round(float(json_annos[anno_index]['frames'][person_frame_index]['rect']['br']['x']) * img_w)
                bottom = round(float(json_annos[anno_index]['frames'][person_frame_index]['rect']['br']['y']) * img_h)
                f.write('person %d %d %d %d\n' % (left, top, right, bottom))
    f.close()
    cv2.imwrite(os.path.join(root_output_img, frame_names[image_id]), image_res)

