import cv2
import os

import numpy
import torch

from detector import YOLOv5Detector
from utils.plots import Annotator, colors
from utils.general import non_max_suppression_tiles
from utils.torch_utils import select_device, time_sync
from tiles_manager import TilesManager


small_object_ratio = 0.05
historical_frames_num = 50
Bayes_data_frames_num = 50
Bayes_update_frequency = 5
decision_pick_threshold = 1
tile_size = 640
device = 0  # cuda:0
source = './data/PANDA8k/scene3/images'
output = './runs/hd'
image_ids = sorted(os.listdir(source))
image_example = cv2.imread(os.path.join(source, image_ids[0]))
image_h, image_w = image_example.shape[0:2]
yolov5_detector_whole = YOLOv5Detector(weights='./yolov5l.pt', device=device)
yolov5_detector_tiles = YOLOv5Detector(weights='./yolov5s.pt', device=device)
tiles_manager = TilesManager(image_example.shape, tile_size, tile_size, small_object_ratio, historical_frames_num, Bayes_data_frames_num)
device = select_device(device)


# Stage 1 - Historical information initialization.
# Full fill the N historical frames.
print('Stage 1')
for frame_id in range(historical_frames_num):
    image = cv2.imread(os.path.join(source, image_ids[frame_id]))
    time_whole_1 = time_sync()
    det_whole = yolov5_detector_whole.detect(image, image_ids[frame_id]).cpu().numpy()
    time_whole_2 = time_sync()
    det_all_tiles = torch.rand(0, 6).to(device)
    time_all_tiles_1 = time_sync()
    for j in range(len(tiles_manager.tiles)):
        # tile = tiles_manager.tiles[j]
        tile_extended = tiles_manager.tile_extension(j)
        image_tile = image[tile_extended[0]:tile_extended[2], tile_extended[1]:tile_extended[3]]
        det_tile = yolov5_detector_tiles.detect(image_tile, str('%s_tile_%d' % (image_ids[frame_id], j)))  # left, top, right, bottom
        det_tile[:, 0] += tile_extended[1]
        det_tile[:, 1] += tile_extended[0]
        det_tile[:, 2] += tile_extended[1]
        det_tile[:, 3] += tile_extended[0]
        det_all_tiles = torch.cat((det_all_tiles, det_tile), 0)
    det_all_tiles = det_all_tiles.cpu().numpy()
    time_all_tiles_2 = time_sync()
    time_nms_cut_1 = time_sync()
    det_all_tiles = tiles_manager.remove_cut_objects(det_all_tiles)
    det_all_tiles = non_max_suppression_tiles(det_all_tiles, yolov5_detector_tiles.iou_thres)
    time_nms_cut_2 = time_sync()
    time_tile_whole_1 = time_sync()
    det_all_tiles = tiles_manager.remove_tile_res_by_whole(det_whole, det_all_tiles)
    time_tile_whole_2 = time_sync()
    time_record_0 = time_sync()
    tiles_manager.objects_num_record(frame_id, det_whole, det_all_tiles)
    time_record_1 = time_sync()
    print('%d, time_whole %.3fs, time_all_tiles %.3fs, time_nms_cut %.3fs, time_tile_whole %.3fs, time_record %.3f' % (frame_id, (time_whole_2 - time_whole_1), (time_all_tiles_2 - time_all_tiles_1), (time_nms_cut_2 - time_nms_cut_1), (time_tile_whole_2 - time_tile_whole_1), (time_record_1 - time_record_0)))


# Stage 2 - Bayes data collecting.
print('Stage 2')
for frame_id in range(historical_frames_num, historical_frames_num + Bayes_data_frames_num):
    image = cv2.imread(os.path.join(source, image_ids[frame_id]))
    time_whole_1 = time_sync()
    det_whole = yolov5_detector_whole.detect(image, image_ids[frame_id]).cpu().numpy()
    time_whole_2 = time_sync()
    det_all_tiles = torch.rand(0, 6).to(device)
    time_all_tiles_1 = time_sync()
    for j in range(len(tiles_manager.tiles)):
        tile_extended = tiles_manager.tile_extension(j)
        image_tile = image[tile_extended[0]:tile_extended[2], tile_extended[1]:tile_extended[3]]
        det_tile = yolov5_detector_tiles.detect(image_tile, str('%s_tile_%d' % (image_ids[frame_id], j)))  # left, top, right, bottom
        det_tile[:, 0] += tile_extended[1]
        det_tile[:, 1] += tile_extended[0]
        det_tile[:, 2] += tile_extended[1]
        det_tile[:, 3] += tile_extended[0]
        det_all_tiles = torch.cat((det_all_tiles, det_tile), 0)
    det_all_tiles = det_all_tiles.cpu().numpy()
    time_all_tiles_2 = time_sync()
    time_nms_cut_1 = time_sync()
    det_all_tiles = tiles_manager.remove_cut_objects(det_all_tiles)
    det_all_tiles = non_max_suppression_tiles(det_all_tiles, yolov5_detector_tiles.iou_thres)
    time_nms_cut_2 = time_sync()
    time_tile_whole_1 = time_sync()
    det_all_tiles = tiles_manager.remove_tile_res_by_whole(det_whole, det_all_tiles)
    time_tile_whole_2 = time_sync()
    time_record_0 = time_sync()
    tiles_manager.objects_num_record(frame_id, det_whole, det_all_tiles)
    time_record_1 = time_sync()
    time_feature_0 = time_sync()
    tiles_manager.compute_features_and_record_Bayes_data(frame_id)
    time_feature_1 = time_sync()
    time_ground_truth_0 = time_sync()
    tiles_manager.compute_binary_ground_truth_and_record_Bayes_data(frame_id)
    time_ground_truth_1 = time_sync()
    print('%d, t_whole %.3f, t_tiles %.3f, t_nms_cut %.3f, t_tile_whole %.3f, t_feature %.3f, t_record %.3f, t_gt %.3f' % (frame_id, time_whole_2-time_whole_1, time_all_tiles_2-time_all_tiles_1, time_nms_cut_2-time_nms_cut_1, time_tile_whole_2-time_tile_whole_1, time_feature_1-time_feature_0, time_record_1-time_record_0, time_ground_truth_1-time_ground_truth_0))


# Stage 3 - Practice.
print('Stage 3')
decision_pick = numpy.zeros(len(tiles_manager.tiles))
for frame_id in range(historical_frames_num + Bayes_data_frames_num, len(image_ids)):
    do_Bayes_update = False
    if (frame_id - (historical_frames_num + Bayes_data_frames_num)) % Bayes_update_frequency == 0:
        do_Bayes_update = True
    image = cv2.imread(os.path.join(source, image_ids[frame_id]))
    time_whole_1 = time_sync()
    det_whole = yolov5_detector_whole.detect(image, image_ids[frame_id]).cpu().numpy()
    time_whole_2 = time_sync()
    det_all_tiles = torch.rand(0, 6).to(device)
    time_all_tiles_1 = time_sync()
    for j in range(len(tiles_manager.tiles)):
        if do_Bayes_update is False and decision_pick[j] <= decision_pick_threshold:
            continue
        tile_extended = tiles_manager.tile_extension(j)
        image_tile = image[tile_extended[0]:tile_extended[2], tile_extended[1]:tile_extended[3]]
        det_tile = yolov5_detector_tiles.detect(image_tile, str('%s_tile_%d' % (image_ids[frame_id], j)))  # left, top, right, bottom
        det_tile[:, 0] += tile_extended[1]
        det_tile[:, 1] += tile_extended[0]
        det_tile[:, 2] += tile_extended[1]
        det_tile[:, 3] += tile_extended[0]
        det_all_tiles = torch.cat((det_all_tiles, det_tile), 0)
    det_all_tiles = det_all_tiles.cpu().numpy()
    time_all_tiles_2 = time_sync()
    time_nms_cut_1 = time_sync()
    det_all_tiles = tiles_manager.remove_cut_objects(det_all_tiles)
    det_all_tiles = non_max_suppression_tiles(det_all_tiles, yolov5_detector_tiles.iou_thres)
    time_nms_cut_2 = time_sync()
    time_tile_whole_1 = time_sync()
    det_all_tiles = tiles_manager.remove_tile_res_by_whole(det_whole, det_all_tiles)
    time_tile_whole_2 = time_sync()
    if do_Bayes_update is False:
        print('%d, t_whole %.3f, t_tiles %.3f, t_nms_cut %.3f, t_tile_whole %.3f' % (frame_id, time_whole_2-time_whole_1, time_all_tiles_2-time_all_tiles_1, time_nms_cut_2-time_nms_cut_1, time_tile_whole_2-time_tile_whole_1))
    if do_Bayes_update is True:
        time_record_0 = time_sync()
        tiles_manager.objects_num_record(frame_id, det_whole, det_all_tiles)
        time_record_1 = time_sync()
        time_feature_0 = time_sync()
        tiles_manager.compute_features_and_record_Bayes_data(frame_id)
        time_feature_1 = time_sync()
        time_ground_truth_0 = time_sync()
        tiles_manager.compute_binary_ground_truth_and_record_Bayes_data(frame_id)
        time_ground_truth_1 = time_sync()
        time_probability_0 = time_sync()
        decision_pick = tiles_manager.Bayes_manager.compute_Bayes_probability_and_pick(frame_id)
        time_probability_1 = time_sync()
        print('%d, t_whole %.3f, t_tiles %.3f, t_nms_cut %.3f, t_tile_whole %.3f, t_feature %.3f, t_record %.3f, t_gt %.3f, t_decision %.3f' % (frame_id, time_whole_2-time_whole_1, time_all_tiles_2-time_all_tiles_1, time_nms_cut_2-time_nms_cut_1, time_tile_whole_2-time_tile_whole_1, time_feature_1-time_feature_0, time_record_1-time_record_0, time_ground_truth_1-time_ground_truth_0, time_probability_1 - time_probability_0))
    # # -----Save detection results------
    # file_name = frame_id
    # # file_name = image_ids[frame_id].split('.')[0]
    # f = open(str('./runs/hd/bboxes/%s.txt' % file_name), 'w')
    # for *xyxy, conf, cls in reversed(det_all_tiles):
    #     if cls != 0:
    #         continue
    #     f.write('person %.3f %d %d %d %d\n' % (conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    # for *xyxy, conf, cls in reversed(det_whole):
    #     if cls != 0:
    #         continue
    #     f.write('person %.3f %d %d %d %d\n' % (conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    # f.close()
    # # -----Draw bboxes on the whole image and save-----
    # save_path = os.path.join(output, str('images/%s.jpg' % file_name))
    # image_res = image.copy()
    # annotator = Annotator(image_res, line_width=3, example=str(yolov5_detector_tiles.model.names))
    # frame_id_loop_Bayes = frame_id % tiles_manager.Bayes_data_frames_num
    # for j in range(len(tiles_manager.tiles)):
    #     # tile_extended = tiles_manager.tiles[j]
    #     # cv2.rectangle(image_res, (tile_extended[1], tile_extended[0]), (tile_extended[3], tile_extended[2]), (0, 255, 255), thickness=3)
    #     tile_extended = tiles_manager.tile_extension(j)
    #     if decision_pick[j] > decision_pick_threshold:
    #         # shrink = 15
    #         shrink = 0
    #         cv2.rectangle(image_res, (tile_extended[1]+shrink, tile_extended[0]+shrink), (tile_extended[3]-shrink, tile_extended[2]-shrink), (0, 255, 255), thickness=30)
    # for *xyxy, conf, cls in reversed(det_all_tiles):
    #     size = max((xyxy[2] - xyxy[0]), (xyxy[3] - xyxy[1]))
    #     # annotator.box_label(xyxy, label=f'{int(cls)} {conf:.2f}', color=(0, 255, 0))
    #     annotator.box_label(xyxy, label=False, color=(0, 255, 0))
    # for *xyxy, conf, cls in reversed(det_whole):
    #     size = max((xyxy[2] - xyxy[0]), (xyxy[3] - xyxy[1]))
    #     # annotator.box_label(xyxy, label=f'{int(cls)} {conf:.2f}', color=(0, 255, 0))
    #     annotator.box_label(xyxy, label=False, color=(0, 0, 255))
    # cv2.imwrite(save_path, image_res)

