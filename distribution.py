import math

import numpy
import os


root_input = './runs/hd/temp/features2_bayes.txt'
stride = 0.2
record = numpy.zeros(math.ceil(1 / stride))
# record = numpy.zeros(200)
record_0 = 0
record_1 = 0


f = open(root_input)
lines = f.readlines()
for i in range(len(lines)):
    line = lines[i].split('\t')[:-1]
    for j in range(len(line)):
        value = float(line[j])
        if value == 0:
            record_0 += 1
        elif value == 1:
            record_1 += 1
        else:
            segment_id = math.floor(value / stride)
            record[segment_id] += 1
f.close()


print('0\t%d' % record_0)
for i in range(len(record)):
    print('%.2f-%.2f\t%d' % (i*stride, (i+1)*stride, record[i]))
print('1\t%d' % record_1)

