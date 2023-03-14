import math
import numpy


class BayesManager:
    def __init__(self, Bayes_data_frames_num, num_tiles):
        self.e = 2.718282
        self.Bayes_data_frames_num = Bayes_data_frames_num
        self.num_tiles = num_tiles
        self.Bayes_samples = numpy.zeros((self.Bayes_data_frames_num, self.num_tiles, 4))  # 3 means word_feature1, word_feature2, word_feature3, ground_truth
        self.high_thres_feature_1_to_word_1 = 0.2
        self.stride_feature_1_to_word_1 = 0.02
        self.stride_feature_2_to_word_2 = 0.2
        self.high_thres_feature_3_to_word_3 = 0.1
        self.stride_feature_3_to_word_3 = 0.02
        self.num_word_1_values = int(self.high_thres_feature_1_to_word_1 / self.stride_feature_1_to_word_1 + 2)
        self.num_word_2_values = int(1 / self.stride_feature_2_to_word_2 + 2)
        self.num_word_3_values = int(self.high_thres_feature_3_to_word_3 / self.stride_feature_3_to_word_3 + 2)
        self.probability_pick = 0
        self.probability_word12_not_pick = numpy.zeros((self.num_word_1_values, self.num_word_2_values))
        self.probability_word12_pick = numpy.zeros((self.num_word_1_values, self.num_word_2_values))
        self.probability_word3_not_pick = numpy.zeros(self.num_word_3_values)
        self.probability_word3_pick = numpy.zeros(self.num_word_3_values)

    def compute_and_record_words_one_frame(self, frame_id, feature_1_one_frame, feature_2_one_frame, feature_3_one_frame):
        frame_id_loop_Bayes = frame_id % self.Bayes_data_frames_num
        for tile_id in range(self.num_tiles):
            feature_1_tile = feature_1_one_frame[tile_id]
            feature_2_tile = 1 / math.pow(self.e, feature_2_one_frame[tile_id])
            feature_3_tile = feature_3_one_frame[tile_id]
            if feature_1_tile == 0:
                word_feature_1 = 0
            elif feature_1_tile >= self.high_thres_feature_1_to_word_1:
                word_feature_1 = (self.high_thres_feature_1_to_word_1 / self.stride_feature_1_to_word_1) + 1
            else:
                word_feature_1 = (feature_1_tile / self.stride_feature_1_to_word_1) + 1
            if feature_2_tile == 0:
                word_feature_2 = 0
            else:
                word_feature_2 = (feature_2_tile / self.stride_feature_2_to_word_2) + 1
            if feature_3_tile == 0:
                word_feature_3 = 0
            elif feature_3_tile >= self.high_thres_feature_3_to_word_3:
                word_feature_3 = (self.high_thres_feature_3_to_word_3 / self.stride_feature_3_to_word_3) + 1
            else:
                word_feature_3 = (feature_3_tile / self.stride_feature_3_to_word_3) + 1
            self.Bayes_samples[frame_id_loop_Bayes][tile_id][0] = word_feature_1
            self.Bayes_samples[frame_id_loop_Bayes][tile_id][1] = word_feature_2
            self.Bayes_samples[frame_id_loop_Bayes][tile_id][2] = word_feature_3

    def record_ground_truth_one_frame(self, frame_id, ground_truth_one_frame):
        frame_id_loop_Bayes = frame_id % self.Bayes_data_frames_num
        for tile_id in range(self.num_tiles):
            self.Bayes_samples[frame_id_loop_Bayes][tile_id][3] = ground_truth_one_frame[tile_id]
        return

    def compute_Bayes_probability_and_pick(self, frame_id):
        num_pick = 0
        num_word12_total = numpy.zeros((self.num_word_1_values, self.num_word_2_values))
        num_word12_pick = numpy.zeros((self.num_word_1_values, self.num_word_2_values))
        num_word3_total = numpy.zeros(self.num_word_3_values)
        num_word3_pick = numpy.zeros(self.num_word_3_values)
        for frame_index in range(self.Bayes_data_frames_num):
            for tile_id in range(self.num_tiles):
                word1, word2, word3, ground_truth = self.Bayes_samples[frame_index][tile_id]
                word1, word2, word3, ground_truth = int(word1), int(word2), int(word3), int(ground_truth)
                num_word12_total[word1][word2] += 1
                num_word3_total[word3] += 1
                if ground_truth == 1:
                    num_pick += 1
                    num_word12_pick[word1][word2] += 1
                    num_word3_pick[word3] += 1
        num_samples = self.Bayes_data_frames_num * self.num_tiles
        self.probability_word12_not_pick = (num_word12_total - num_word12_pick) / (num_samples - num_pick)
        self.probability_word12_pick = num_word12_pick / num_pick
        self.probability_word3_not_pick = (num_word3_total - num_word3_pick) / (num_samples - num_pick)
        self.probability_word3_pick = num_word3_pick / num_pick
        self.probability_pick = num_pick / num_samples
        decision_pick = numpy.zeros(self.num_tiles)
        frame_id_loop_Bayes = frame_id % self.Bayes_data_frames_num
        for tile_id in range(self.num_tiles):
            word1, word2, word3, ground_truth = self.Bayes_samples[frame_id_loop_Bayes][tile_id]
            word1, word2, word3, ground_truth = int(word1), int(word2), int(word3), int(ground_truth)
            if self.probability_word12_not_pick[word1][word2] == 0 or self.probability_word3_not_pick[word3] == 0 or (1 - self.probability_pick) == 0:
                decision_pick[tile_id] = (self.probability_word12_pick[word1][word2] * self.probability_word3_pick[word3] * self.probability_pick) / 1e-6
            else:
                decision_pick[tile_id] = (self.probability_word12_pick[word1][word2] * self.probability_word3_pick[word3] * self.probability_pick) / (self.probability_word12_not_pick[word1][word2] * self.probability_word3_not_pick[word3] * (1 - self.probability_pick))
        return decision_pick
