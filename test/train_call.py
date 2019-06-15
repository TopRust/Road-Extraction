modelpy = \
"./script/chain.py"
gpu_id = 0
fliplr = True
rotate = True
norm = True
image_side = 64
label_side = 16
out_directory = \
'./road_resullt_2'
epoch = 0
from script.train import train

train(modelpy, gpu_id, fliplr, rotate, norm, image_side, label_side, out_directory, epoch)
