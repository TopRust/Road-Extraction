
import argparse
import glob
import os
import shutil
import time

import numpy as np

import cv2 as cv
import lmdb

# crop a image into patches
def divide_to_patches(stride, sat_size, map_size, sat_im, map_im):

    sat_patches = []
    map_patches = []

    for x in range(0, sat_im.shape[0] - sat_size, stride):
        for y in range(0, sat_im.shape[1] - sat_size, stride):

            if (x + sat_size > sat_im.shape[0]):
                x = sat_im.shape[0]- sat_size

            if (y + sat_size > sat_im.shape[1]):
                 y = sat_im.shape[1] - sat_size

            sat_patch = sat_im[x: x + sat_size, y: y + sat_size]
            map_patch = map_im[x + sat_size // 2 - map_size // 2: x + sat_size // 2 + map_size // 2,
                                y + sat_size // 2 - map_size // 2: y + sat_size // 2 + map_size // 2]
            
            sat_patches.append(sat_patch)
            map_patches.append(map_patch)

    return sat_patches, map_patches

# create building and road multi-class dataset
def create_merged_map():
    
    # copy sat images
    for data_type in ['train', 'test', 'valid']:

        out_dir = 'data/mass_merged/%s/sat' % data_type

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for fn in glob.glob('data/mass_buildings/%s/sat/*.tiff' % data_type):
            shutil.copy(fn, '%s/%s' % (out_dir, os.path.basename(fn)))

    road_maps = dict([(os.path.basename(fn).split('.')[0], fn)
                      for fn in glob.glob('data/mass_roads/*/map/*.tif')])

    # combine map images
    for data_type in ['train', 'test', 'valid']:

        out_dir = 'data/mass_merged/%s/map' % data_type

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for fn in glob.glob('data/mass_buildings/%s/map/*.tif' % data_type):

            base = os.path.basename(fn).split('.')[0]

            building_map = cv.imread(fn, cv.IMREAD_GRAYSCALE)
            road_map = cv.imread(road_maps[base], cv.IMREAD_GRAYSCALE)
            _, building_map = cv.threshold(building_map, 0, 1, cv.THRESH_BINARY)
            _, road_map = cv.threshold(road_map, 0, 1, cv.THRESH_BINARY)

            h, w = road_map.shape
            merged_map = np.zeros((h, w))   # background is 0
            merged_map += building_map  # building is 1
            merged_map += road_map * 2  # road is 2
            merged_map = np.where(merged_map > 2, 2, merged_map)
            cv.imwrite('data/mass_merged/%s/map/%s.tif' % (data_type, base),
                       merged_map)
            print(merged_map.shape, fn)
            # create a new merged_map, 0 to (1, 0, 0) pixel is background, 1 to (0, 1, 0) pixel is building, 2 to (0, 0, 1) pixel is road
            merged_map = np.array([np.where(merged_map == 0, 1, 0),
                                   np.where(merged_map == 1, 1, 0),
                                   np.where(merged_map == 2, 1, 0)])
            # equal to merged_map.transpose(1, 2, 0)
            merged_map = merged_map.swapaxes(0, 2).swapaxes(0, 1) 
            cv.imwrite('data/mass_merged/%s/map/%s.png' % (data_type, base),
                       merged_map * 255)

# save create patches into lmdb dataset
def create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                   sat_data_dir, map_data_dir, sat_out_dir, map_out_dir):

    if os.path.exists(sat_out_dir):
        shutil.rmtree(sat_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    os.makedirs(sat_out_dir)
    os.makedirs(map_out_dir)

    # db
    sat_env = lmdb.Environment(sat_out_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=False)
    map_env = lmdb.Environment(map_out_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=False)

    # patch size
    sat_size = sat_patch_size
    map_size = map_patch_size
    print('patch size:', sat_size, map_size, stride)

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))
    index = np.arange(len(sat_fns))

    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    n_all_files = len(sat_fns)
    print('n_all_files:', n_all_files)

    n_patches = 0
    for i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):

        if ((os.path.basename(sat_fn).split('.')[0]) !=
                (os.path.basename(map_fn).split('.')[0])):
            print('File names are different', sat_fn, map_fn)
            return

        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        map_im = map_im[:, :, np.newaxis]

        st = time.time()
        sat_patches, map_patches = divide_to_patches(
            stride, sat_size, map_size, sat_im, map_im)
        print('divide: {}'.format(time.time() - st))

        sat_patches = np.asarray(sat_patches, dtype=np.uint8)
        map_patches = np.asarray(map_patches, dtype=np.uint8)

        for patch_i in range(sat_patches.shape[0]):

            sat_patch = sat_patches[patch_i]
            map_patch = map_patches[patch_i]

            bn = str(n_patches).encode()
            sat_txn.put(bn, sat_patch.tobytes())
            map_txn.put(bn, map_patch.tobytes())

            n_patches += 1

        print(i, '/', n_all_files, 'n_patches:', n_patches)
    
    sat_txn.commit()
    sat_env.close()
    map_txn.commit()
    map_env.close()
    print('patches:\t', n_patches)

