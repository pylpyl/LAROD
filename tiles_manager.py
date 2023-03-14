import math
import numpy
from Bayes_manager import BayesManager


class TilesManager:
    def __init__(self, img_shape, tile_h, tile_w, small_object_ratio, historical_frames_num, Bayes_data_frames_num):
        self.img_w = img_shape[1]
        self.img_h = img_shape[0]
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.segments_h = math.ceil(self.img_h / self.tile_h)
        self.segments_w = math.ceil(self.img_w / self.tile_w)
        self.tiles_value_modification_rate = 0
        self.tiles = self.tile_split_init()
        self.overlap = 0
        self.small_object_ratio = small_object_ratio
        self.remove_cut_object_close_threshold = 0
        self.accidental_small_object_close_ratio = 0.01
        self.accidental_small_object_difference_ratio = 0.5
        self.historical_frames_num = historical_frames_num
        self.Bayes_data_frames_num = Bayes_data_frames_num
        self.num_small_obj_total_per_frame = numpy.zeros(self.historical_frames_num)
        self.num_small_obj_tiles_per_frame = numpy.zeros((self.historical_frames_num, self.segments_h * self.segments_w))
        self.num_obj_tiles_per_frame = numpy.zeros((self.historical_frames_num, self.segments_h * self.segments_w))
        self.feature_1 = numpy.zeros((self.Bayes_data_frames_num, self.segments_h * self.segments_w))
        self.feature_2 = numpy.zeros((self.Bayes_data_frames_num, self.segments_h * self.segments_w))
        self.feature_3 = numpy.zeros((self.Bayes_data_frames_num, self.segments_h * self.segments_w))
        self.max_small_obj_size_tiles_per_frame = numpy.zeros((self.historical_frames_num, self.segments_h * self.segments_w))
        self.ground_truth_true_small_obj_ratio_thres = 0.02
        self.ground_truth_tiles_per_frame = numpy.zeros((self.Bayes_data_frames_num, self.segments_h * self.segments_w))
        self.Bayes_manager = BayesManager(self.Bayes_data_frames_num, self.segments_h * self.segments_w)

    def tile_split_init(self):
        tiles = []  # (top, left, bottom, right)
        for i in range(self.segments_h):
            for j in range(self.segments_w):
                top_loc = i * self.tile_h
                left_loc = j * self.tile_w
                if top_loc < self.img_h-1 and left_loc < self.img_w-1:
                    tiles.append([top_loc, left_loc, min(top_loc+self.tile_h-1, self.img_h), min(left_loc+self.tile_w-1, self.img_w)])
        return tiles

    def remove_tile_res_by_whole(self, det_whole, det_all_tiles):
        cut_ids = []
        for i in reversed(range(len(det_all_tiles))):
            center_horizontal = (det_all_tiles[i][2] + det_all_tiles[i][0]) / 2
            center_vertical = (det_all_tiles[i][3] + det_all_tiles[i][1]) / 2
            for j in range(len(det_whole)):
                if det_whole[j][0] <= center_horizontal <= det_whole[j][2] and det_whole[j][1] <= center_vertical <= \
                        det_whole[j][3]:
                    cut_ids.append(i)
                    break
        det_all_tiles = numpy.delete(det_all_tiles, cut_ids, axis=0)
        return det_all_tiles

    def remove_cut_objects(self, det_all_tiles):
        cut_ids = []
        for i in range(len(det_all_tiles)):
            # print('-----------------------------------------')
            det_left = det_all_tiles[i][0]
            det_top = det_all_tiles[i][1]
            det_right = det_all_tiles[i][2]
            det_bottom = det_all_tiles[i][3]
            center_horizontal = (det_left + det_right) / 2
            center_vertical = (det_top + det_bottom) / 2
            # print('res center %d %d' % (center_horizontal, center_vertical))
            col_tile_left = math.floor((center_horizontal - math.floor(self.tile_w/2)) / self.tile_w)
            col_tile_right = col_tile_left + 1
            row_tile_top = math.floor((center_vertical - math.floor(self.tile_h/2)) / self.tile_h)
            row_tile_bottom = row_tile_top + 1
            # print('neighbor tile %d %d %d %d' % (row_tile_top, col_tile_left, row_tile_bottom, col_tile_right))
            if col_tile_left >= 0 and row_tile_top >= 0:
                tile_id_tl = self.convert_row_col_to_tile_id(row_tile_top, col_tile_left)
                tile_tl_extended = self.tile_extension(tile_id_tl)
                if abs(det_bottom - tile_tl_extended[2]) <= self.remove_cut_object_close_threshold or abs(det_right - tile_tl_extended[3]) <= self.remove_cut_object_close_threshold:
                    cut_ids.append(i)
                    continue
            if col_tile_right <= self.segments_w-1 and row_tile_top >= 0:
                tile_id_tr = self.convert_row_col_to_tile_id(row_tile_top, col_tile_right)
                tile_tr_extended = self.tile_extension(tile_id_tr)
                if abs(det_bottom - tile_tr_extended[2]) <= self.remove_cut_object_close_threshold or abs(det_left - tile_tr_extended[1]) <= self.remove_cut_object_close_threshold:
                    cut_ids.append(i)
                    continue
            if col_tile_left >= 0 and row_tile_bottom <= self.segments_h-1:
                tile_id_bl = self.convert_row_col_to_tile_id(row_tile_bottom, col_tile_left)
                tile_bl_extended = self.tile_extension(tile_id_bl)
                if abs(det_top - tile_bl_extended[0]) <= self.remove_cut_object_close_threshold or abs(det_right - tile_bl_extended[3]) <= self.remove_cut_object_close_threshold:
                    cut_ids.append(i)
                    continue
            if col_tile_right <= self.segments_w-1 and row_tile_bottom <= self.segments_h-1:
                tile_id_br = self.convert_row_col_to_tile_id(row_tile_bottom, col_tile_right)
                tile_br_extended = self.tile_extension(tile_id_br)
                if abs(det_top - tile_br_extended[0]) <= self.remove_cut_object_close_threshold or abs(det_left - tile_br_extended[1]) <= self.remove_cut_object_close_threshold:
                    cut_ids.append(i)
                    continue
            # print('--------------------------')
            # print('res center %d %d' % (center_horizontal, center_vertical))
            # print('neighbor tile %d %d %d %d' % (row_tile_top, col_tile_left, row_tile_bottom, col_tile_right))
        det_all_tiles = numpy.delete(det_all_tiles, cut_ids, axis=0)
        return det_all_tiles

    # def remove_accidental_small_objects(self, det_whole, det_all_tiles):
    #     accidental_small_object_ids_in_whole = []
    #     accidental_small_object_ids_in_all_tiles = []
    #     small_obj_thres = self.img_h * self.small_object_ratio
    #     det_total_list = numpy.concatenate((det_whole, det_all_tiles))
    #     num_whole = len(det_whole)
    #     for i in range(len(det_total_list)):
    #         width_i = det_total_list[i][2] - det_total_list[i][0]
    #         height_i = det_total_list[i][3] - det_total_list[i][1]
    #         size_i = max(width_i, height_i)
    #         if size_i <= small_obj_thres:
    #             continue
    #         for j in range(len(det_total_list)):
    #             if i == j:
    #                 continue
    #             if j < num_whole and j in accidental_small_object_ids_in_whole:
    #                 continue
    #             if j >= num_whole and (j-num_whole) in accidental_small_object_ids_in_all_tiles:
    #                 continue
    #             width_j = det_total_list[j][2] - det_total_list[j][0]
    #             height_j = det_total_list[j][3] - det_total_list[j][1]
    #             size_j = max(width_j, height_j)
    #             if size_j > small_obj_thres:
    #                 continue
    #             if size_j >= size_i * self.accidental_small_object_difference_ratio:
    #                 continue
    #             center_i_horizontal = round((det_total_list[i][2] + det_total_list[i][0]) / 2)
    #             center_i_vertical = round((det_total_list[i][3] + det_total_list[i][1]) / 2)
    #             center_j_horizontal = round((det_total_list[j][2] + det_total_list[j][0]) / 2)
    #             center_j_vertical = round((det_total_list[j][3] + det_total_list[j][1]) / 2)
    #             if abs(center_i_vertical-center_j_vertical) <= ((height_i+height_j)/2)*(1+self.accidental_small_object_close_ratio) and abs(center_i_horizontal-center_j_horizontal) <= ((width_i+width_j)/2)*(1+self.accidental_small_object_close_ratio):
    #                 if j < num_whole:
    #                     accidental_small_object_ids_in_whole.append(j)
    #                 else:
    #                     accidental_small_object_ids_in_all_tiles.append(j-num_whole)
    #     det_whole = numpy.delete(det_whole, accidental_small_object_ids_in_whole, axis=0)
    #     det_all_tiles = numpy.delete(det_all_tiles, accidental_small_object_ids_in_all_tiles, axis=0)
    #     return det_whole, det_all_tiles

    def convert_location_to_tile_id(self, location_horizontal, location_vertical):
        tile_row = math.floor(location_vertical / self.tile_h)
        tile_col = math.floor(location_horizontal / self.tile_w)
        tile_id = tile_row * self.segments_w + tile_col
        return tile_id

    def convert_tile_id_to_row_col(self, tile_id):
        tile_row = math.floor(tile_id / self.segments_w)
        tile_col = tile_id % self.segments_w
        return tile_row, tile_col

    def convert_row_col_to_tile_id(self, tile_row, tile_col):
        tile_id = tile_row * self.segments_w + tile_col
        return tile_id

    def neighbor_tiles_id(self, tile_id):
        neighbors = []
        tile_row, tile_col = self.convert_tile_id_to_row_col(tile_id)
        if tile_row >= 1:
            neighbors.append(self.convert_row_col_to_tile_id(tile_row-1, tile_col))
        if tile_row < self.segments_h - 1:
            neighbors.append(self.convert_row_col_to_tile_id(tile_row+1, tile_col))
        if tile_col >= 1:
            neighbors.append(self.convert_row_col_to_tile_id(tile_row, tile_col-1))
        if tile_col < self.segments_w - 1:
            neighbors.append(self.convert_row_col_to_tile_id(tile_row, tile_col+1))
        return neighbors

    def objects_num_record(self, frame_id, det_whole, det_all_tiles):
        frame_id_loop = frame_id % self.historical_frames_num
        small_obj_thres = self.img_h * self.small_object_ratio
        self.num_small_obj_total_per_frame[frame_id_loop] = 0
        self.num_small_obj_tiles_per_frame[frame_id_loop] = numpy.zeros(self.segments_h * self.segments_w)
        self.num_obj_tiles_per_frame[frame_id_loop] = numpy.zeros(self.segments_h * self.segments_w)
        self.max_small_obj_size_tiles_per_frame[frame_id_loop] = numpy.zeros(self.segments_h * self.segments_w)
        for i in range(len(det_whole)):
            center_horizontal = round((det_whole[i][2] + det_whole[i][0]) / 2)
            center_vertical = round((det_whole[i][3] + det_whole[i][1]) / 2)
            tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
            size = max((det_whole[i][2]-det_whole[i][0]), (det_whole[i][3]-det_whole[i][1]))
            self.num_obj_tiles_per_frame[frame_id_loop][tile_id] += 1
            if size <= small_obj_thres:
                self.num_small_obj_total_per_frame[frame_id_loop] += 1
                self.num_small_obj_tiles_per_frame[frame_id_loop][tile_id] += 1
                self.max_small_obj_size_tiles_per_frame[frame_id_loop][tile_id] = max(self.max_small_obj_size_tiles_per_frame[frame_id_loop][tile_id], size)
        for i in range(len(det_all_tiles)):
            center_horizontal = round((det_all_tiles[i][2] + det_all_tiles[i][0]) / 2)
            center_vertical = round((det_all_tiles[i][3] + det_all_tiles[i][1]) / 2)
            tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
            size = max((det_all_tiles[i][2]-det_all_tiles[i][0]), (det_all_tiles[i][3]-det_all_tiles[i][1]))
            self.num_obj_tiles_per_frame[frame_id_loop][tile_id] += 1
            if size <= small_obj_thres:
                self.num_small_obj_total_per_frame[frame_id_loop] += 1
                self.num_small_obj_tiles_per_frame[frame_id_loop][tile_id] += 1
                self.max_small_obj_size_tiles_per_frame[frame_id_loop][tile_id] = max(self.max_small_obj_size_tiles_per_frame[frame_id_loop][tile_id], size)
        return

    def tile_extension(self, tile_id):
        extension_size = math.floor(numpy.max(self.max_small_obj_size_tiles_per_frame[:, tile_id]) / 2)
        # extension_size = 0
        tile_top_extended = max(0, self.tiles[tile_id][0] - extension_size)
        tile_left_extended = max(0, self.tiles[tile_id][1] - extension_size)
        tile_bottom_extended = min(self.img_h-1, self.tiles[tile_id][2] + extension_size)
        tile_right_extended = min(self.img_w-1, self.tiles[tile_id][3] + extension_size)
        return [tile_top_extended, tile_left_extended, tile_bottom_extended, tile_right_extended]

    def compute_features_and_record_Bayes_data(self, frame_id):
        frame_id_loop = frame_id % self.Bayes_data_frames_num
        num_small_obj_total_accumulative = self.num_small_obj_total_per_frame.sum()
        num_small_obj_tiles_accumulative = self.num_small_obj_tiles_per_frame.sum(axis=0)
        feature_1 = num_small_obj_tiles_accumulative / num_small_obj_total_accumulative
        feature_2 = numpy.zeros((self.historical_frames_num, self.segments_h * self.segments_w))
        for frame_index in range(self.historical_frames_num):
            for tile_id in range(len(self.tiles)):
                if self.num_small_obj_total_per_frame[frame_index] == 0:
                    continue
                feature_2[frame_index][tile_id] = self.num_small_obj_tiles_per_frame[frame_index][tile_id] / self.num_small_obj_total_per_frame[frame_index]
        feature_2 = numpy.var(feature_2, axis=0)
        for tile_id in range(len(self.tiles)):
            f_1_temp = feature_1[tile_id]
            self.feature_1[frame_id_loop][tile_id] = f_1_temp
            if f_1_temp == 0:
                self.feature_2[frame_id_loop][tile_id] = 0
            else:
                self.feature_2[frame_id_loop][tile_id] = math.sqrt(feature_2[tile_id]) / f_1_temp
            f_3 = 0
            neighbor_tiles_id = self.neighbor_tiles_id(tile_id)
            for neighbor_tile_id in neighbor_tiles_id:
                f_3 += feature_1[neighbor_tile_id]
            self.feature_3[frame_id_loop][tile_id] = f_3 / len(neighbor_tiles_id)
        self.Bayes_manager.compute_and_record_words_one_frame(frame_id, self.feature_1[frame_id_loop], self.feature_2[frame_id_loop], self.feature_3[frame_id_loop])

    def compute_binary_ground_truth_and_record_Bayes_data(self, frame_id):
        frame_id_loop_Bayes = frame_id % self.Bayes_data_frames_num
        frame_id_loop_history = frame_id % self.historical_frames_num
        for tile_id in range(len(self.tiles)):
            ratio_small_obj_num_tile_over_total_a_frame = self.num_small_obj_tiles_per_frame[frame_id_loop_history][tile_id] / self.num_small_obj_total_per_frame[frame_id_loop_history]
            if ratio_small_obj_num_tile_over_total_a_frame >= self.ground_truth_true_small_obj_ratio_thres:
                self.ground_truth_tiles_per_frame[frame_id_loop_Bayes][tile_id] = 1
            else:
                self.ground_truth_tiles_per_frame[frame_id_loop_Bayes][tile_id] = 0
        self.Bayes_manager.record_ground_truth_one_frame(frame_id, self.ground_truth_tiles_per_frame[frame_id_loop_Bayes])

    def compute_Bayes_probability(self):
        # f = open('./runs/hd/gt.txt', 'w')
        # for tile_id in range(self.segments_h * self.segments_w):
        #     tile_row, tile_col = self.convert_tile_id_to_row_col(tile_id)
        #     f.write('%d-%d' % (tile_row, tile_col))
        #     for frame_id_in_loop in range(self.historical_frames_num):
        #         f.write('\t%d' % self.ground_truth_tiles_per_frame[frame_id_in_loop][tile_id])
        #     f.write('\n')
        # f.close()
        probability_is_true = numpy.sum(self.ground_truth_tiles_per_frame) / (self.historical_frames_num * self.segments_w * self.segments_h)
        print(probability_is_true)

    # def objects_count_for_tiles_accumulative(self, det_whole, det_all_tiles):
    #     det_whole_list = det_whole[:, 0:4].tolist()
    #     det_all_tiles_list = det_all_tiles.tolist()
    #     small_obj_thres = self.img_h * self.small_object_ratio
    #     for i in range(len(det_whole_list)):
    #         center_horizontal = round((det_whole_list[i][2] + det_whole_list[i][0]) / 2)
    #         center_vertical = round((det_whole_list[i][3] + det_whole_list[i][1]) / 2)
    #         tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
    #         size = max((det_whole_list[i][2]-det_whole_list[i][0]), (det_whole_list[i][3]-det_whole_list[i][1]))
    #         self.tiles[tile_id][7] += 1
    #         self.tiles[tile_id][8] = max(self.tiles[tile_id][8], size)
    #         self.tiles[tile_id][9] = min(self.tiles[tile_id][9], size)
    #         if size <= small_obj_thres:
    #             self.tiles[tile_id][5] += 1
    #             self.tiles[tile_id][6] = max(self.tiles[tile_id][6], size)
    #     for i in range(len(det_all_tiles_list)):
    #         center_horizontal = round((det_all_tiles_list[i][2] + det_all_tiles_list[i][0]) / 2)
    #         center_vertical = round((det_all_tiles_list[i][3] + det_all_tiles_list[i][1]) / 2)
    #         tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
    #         size = max((det_all_tiles_list[i][2]-det_all_tiles_list[i][0]), (det_all_tiles_list[i][3]-det_all_tiles_list[i][1]))
    #         self.tiles[tile_id][7] += 1
    #         self.tiles[tile_id][8] = max(self.tiles[tile_id][8], size)
    #         self.tiles[tile_id][9] = min(self.tiles[tile_id][9], size)
    #         if size <= small_obj_thres:
    #             self.tiles[tile_id][5] += 1
    #             self.tiles[tile_id][6] = max(self.tiles[tile_id][6], size)
    #     return
    #
    # def objects_count_for_tiles_one_frame(self, det_whole, det_all_tiles):
    #     det_whole_list = det_whole[:, 0:4].tolist()
    #     det_all_tiles_list = det_all_tiles.tolist()
    #     small_obj_thres = self.img_h * self.small_object_ratio
    #     for i in range(len(self.tiles)):
    #         self.tiles[i][5:10] = [0, 0, 0, 0, self.img_w+1]
    #     for i in range(len(det_whole_list)):
    #         center_horizontal = round((det_whole_list[i][2] + det_whole_list[i][0]) / 2)
    #         center_vertical = round((det_whole_list[i][3] + det_whole_list[i][1]) / 2)
    #         tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
    #         size = max((det_whole_list[i][2]-det_whole_list[i][0]), (det_whole_list[i][3]-det_whole_list[i][1]))
    #         self.tiles[tile_id][7] += 1
    #         self.tiles[tile_id][8] = max(self.tiles[tile_id][8], size)
    #         self.tiles[tile_id][9] = min(self.tiles[tile_id][9], size)
    #         if size <= small_obj_thres:
    #             self.tiles[tile_id][5] += 1
    #             self.tiles[tile_id][6] = max(self.tiles[tile_id][6], size)
    #     for i in range(len(det_all_tiles_list)):
    #         center_horizontal = round((det_all_tiles_list[i][2] + det_all_tiles_list[i][0]) / 2)
    #         center_vertical = round((det_all_tiles_list[i][3] + det_all_tiles_list[i][1]) / 2)
    #         tile_id = self.convert_location_to_tile_id(center_horizontal, center_vertical)
    #         size = max((det_all_tiles_list[i][2]-det_all_tiles_list[i][0]), (det_all_tiles_list[i][3]-det_all_tiles_list[i][1]))
    #         self.tiles[tile_id][7] += 1
    #         self.tiles[tile_id][8] = max(self.tiles[tile_id][8], size)
    #         self.tiles[tile_id][9] = min(self.tiles[tile_id][9], size)
    #         if size <= small_obj_thres:
    #             self.tiles[tile_id][5] += 1
    #             self.tiles[tile_id][6] = max(self.tiles[tile_id][6], size)
    #     return


    # def cutting_exclude_small_obj_record(self, det_all_tiles):
    #     threshold_close = 5
    #     for idx in reversed(range(len(det_all_tiles))):
    #         left_det, top_det, right_det, bottom_det = det_all_tiles[idx][0:4].tolist()
    #         center_horizontal = round((left_det + right_det) / 2)
    #         center_vertical = round((top_det + bottom_det) / 2)
    #         row_tile = math.ceil(center_vertical / self.tile_h)
    #         col_tile = math.ceil(center_horizontal / self.tile_w)
    #         id_tile = self.segments_w * (row_tile - 1) + col_tile - 1
    #         top_tile, left_tile, bottom_tile, right_tile = self.tiles[id_tile][0:4]
    #         height_obj = abs(bottom_det - top_det)
    #         if height_obj>=self.img_h*self.small_object_ratio or abs(top_det-top_tile)<=threshold_close or abs(top_det-bottom_tile)<=threshold_close or abs(bottom_det-top_tile)<=threshold_close or abs(bottom_det-bottom_tile)<=threshold_close or abs(left_det-left_tile)<=threshold_close or abs(left_det-right_tile)<=threshold_close or abs(right_det-left_tile)<=threshold_close or abs(right_det-right_tile)<=threshold_close:
    #             # Ignore cut bboxes. Save uncut and small bboxes only.
    #             det_all_tiles = det_all_tiles[torch.arange(det_all_tiles.size(0)) != idx]
    #             self.tiles[id_tile][5] = max(height_obj, self.tiles[id_tile][5])
    #     return det_all_tiles

    # def update_values_tiles(self, bbox_num_in_tiles):
    #     num_tiles = len(bbox_num_in_tiles)
    #     for i in range(num_tiles):
    #         bbox_num = bbox_num_in_tiles[i]
    #         increase_value = bbox_num * self.tiles_value_modification_rate
    #         self.tiles[i][4] += increase_value
    #     # Ensure the sum is 1
    #     value_total = 0
    #     for i in range(num_tiles):
    #         value_total += self.tiles[i][4]
    #     for i in range(num_tiles):
    #         self.tiles[i][4] = self.tiles[i][4] / value_total


