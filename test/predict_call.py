

gpu = 0
model = \

param = \

test_sat_dir = \

sat_size = 64
map_size = 16
channels = 3
offset = 1
batchsize = 128

from script.predict import predict
predict(gpu, model, param, test_sat_dir, sat_size, map_size, channels, offset, batchsize):

