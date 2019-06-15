offset = 8
channel = 3
pad = 24
relax = 3
steps = 1024
n_thread = 8

epoch = 400 
map_dir = \
'data/mass_merged/test/map'
result_directory = \
'road_result'

from script.evaluate import evaluate
evaluate(offset, channel, pad, relax, steps, n_thread, epoch, map_dir, result_directory)
