n_process=8

label_dir = \
'./data/mass_merged/test/map/'
result_dir = \
'./road_result/'
epoch=10

pad=24
offset=1
channel=3
steps=256
relax=3

from script.evaluate import evaluate
evaluate(n_process, label_dir, result_dir, epoch, pad, offset, channel, steps, relax)

