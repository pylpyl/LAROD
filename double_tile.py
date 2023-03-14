import cv2
import math


class DoubleTile():
    def __init__(self, img_shape, tile_h, tile_w):
        self.img_w = img_shape[1]
        self.img_h = img_shape[0]
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.double_tile_offset = round(self.tile_h/2)
        self.segments_h = math.ceil(self.img_h / self.tile_h)
        self.segments_w = math.ceil(self.img_w / self.tile_w)
        self.tiles_value_modification_rate = 0
        self.tiles_1, self.tiles_2 = self.tile_split_init()

    def tile_split_init(self):
        tiles_1 = []  # (top, left, bottom, right, value)
        tiles_2 = []
        for i in range(self.segments_h):
            for j in range(self.segments_w):
                top_loc_1 = i * self.tile_h
                left_loc_1 = j * self.tile_w
                if top_loc_1 < self.img_h-1 and left_loc_1 < self.img_w-1:
                    tiles_1.append([top_loc_1, left_loc_1, min(top_loc_1+self.tile_h-1, self.img_h), min(left_loc_1+self.tile_w-1, self.img_w), 0])
                if top_loc_1 + self.double_tile_offset < self.img_h-1 and left_loc_1 + self.double_tile_offset < self.img_w-1:
                    top_loc_2 = top_loc_1 + self.double_tile_offset
                    left_loc_2 = left_loc_1 + self.double_tile_offset
                    tiles_2.append([top_loc_2, left_loc_2, min(top_loc_2+self.tile_h-1, self.img_h), min(left_loc_2+self.tile_w-1, self.img_w), 0])
        len_tiles_1 = len(tiles_1)
        for i in range(len_tiles_1):
            tiles_1[i][4] = 1/len_tiles_1
        len_tiles_2 = len(tiles_2)
        for i in range(len_tiles_2):
            tiles_2[i][4] = 1 / len_tiles_2
        self.tiles_value_modification_rate = 1/(10*len_tiles_1)
        return tiles_1, tiles_2

    def update_values_tiles_1(self, bbox_num_in_tiles_1):
        num_tiles_1 = len(bbox_num_in_tiles_1)
        for i in range(num_tiles_1):
            bbox_num = bbox_num_in_tiles_1[i]
            increase_value = bbox_num * self.tiles_value_modification_rate
            self.tiles_1[i][4] += increase_value
        # Ensure the sum is 1
        value_total = 0
        for i in range(num_tiles_1):
            value_total += self.tiles_1[i][4]
        for i in range(num_tiles_1):
            self.tiles_1[i][4] = self.tiles_1[i][4] / value_total


