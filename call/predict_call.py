

gpu = 0
model = \
'./script/chain.py'
epoch = 1
result_dir =\
'./road_result'
test_sat_dir = \
'./data/mass_merged/test/sat'
sat_size = 64
map_size = 16
channels = 3
offset = 1
batchsize = 128

param = \
'{}/model_epoch-{}'.format(result_dir, epoch)

from script.predict import predict
predict(gpu, model, param, test_sat_dir, sat_size, map_size, channels, offset, batchsize)

