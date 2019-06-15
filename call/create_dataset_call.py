from script.create_dataset import create_merged_map, create_patches

sat_patch_size = \
92
map_patch_size = \
24
stride = \
16
map_ch = \
3

sat_directory = \
'data/mass_merged/{}/sat'
map_directory = \
'data/mass_merged/{}/map'
sat_lmdb_directory = \
'data/mass_merged/lmdb/{}_sat'
map_lmdb_directory = \
'data/mass_merged/lmdb/{}_map'

data_type = ['train', 'valid', 'test']

create_merged_map()

for type in data_type:

    create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                    sat_directory.format(type),
                    map_directory.format(type),
                    sat_lmdb_directory.format(type),
                    map_lmdb_directory.format(type))
