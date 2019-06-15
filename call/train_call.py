
n_process = 4
modelpy = \
"./script/chain.py"
gpu_id = -1
fliplr = True
rotate = True
norm = True
image_side = 64
label_side = 16
out_directory = \
'./road_result'
epoch = 0
from script.train import train

train(n_process, modelpy, gpu_id, fliplr, rotate, norm, image_side, label_side, out_directory, epoch)
