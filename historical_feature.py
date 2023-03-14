import os
import numpy


root_input = './runs/hd/res_each_frame'
txt_names = sorted(os.listdir(root_input))

feature_number = numpy.zeros((84, 100))
feature_size = numpy.zeros((84, 100))

for txt_id in range(100):
    f = open(os.path.join(root_input, txt_names[txt_id]))
    lines = f.readlines()
    f.close()
    for line_id in range(len(lines)):
        line = lines[line_id].split(' ')
        small_obj_num = int(line[2])
        obj_num = int(line[4])
        if obj_num == 0:
            continue
        feature_number[line_id][txt_id] = small_obj_num/obj_num

for i in range(len(feature_number)):
    print('%d %.3f' % (i+1, numpy.var(feature_number[i]).item()))

# f = open('./runs/hd/his.txt', 'w')
# for i in range(len(feature_number)):
#     for j in range(100):
#         f.write('%.3f\t' % feature_number[i][j])
#     f.write('\n')
# f.close()





