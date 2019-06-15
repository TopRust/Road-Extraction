# prediction with offsets using a single model

import argparse
import ctypes
import glob
import imp
import os
import re
import time
from multiprocessing import Array
from multiprocessing import Process
from multiprocessing import Queue

import numpy as np
import chainer
from chainer import Variable
from chainer import cuda
from chainer import serializers

import cv2 as cv

# create minibatch
def create_minibatch(sat_size, map_size, offset, h_limit, w_limit, batchsize, ortho, queue):

    for d in range(0, map_size // 2, (map_size // 2) // offset):
        minibatch = []
        for y in range(d, h_limit, map_size):
            for x in range(d, w_limit, map_size):
                if (((y + sat_size) > h_limit) or
                        ((x + sat_size) > w_limit)):
                    break
                # ortho patch
                o_patch = ortho[
                    y:y + sat_size, x:x + sat_size, :].astype(
                    np.float32, copy=False)
                o_patch -= o_patch.reshape(-1, 3).mean(axis=0)
                o_patch /= o_patch.reshape(-1, 3).std(axis=0) + 1e-5
                o_patch = o_patch.transpose((2, 0, 1))

                minibatch.append(o_patch)
                if len(minibatch) == batchsize:
                    queue.put(np.asarray(minibatch, dtype=np.float32))
                    minibatch = []
        queue.put(np.asarray(minibatch, dtype=np.float32))
    queue.put(None)

# tile patches into an image
def tile_patches(sat_size, map_size, offset, h_limit, w_limit, canvas, queue):

    for d in range(0, map_size // 2, (map_size // 2) // offset):
        st = time.time()
        for y in range(d, h_limit, map_size):
            for x in range(d, w_limit, map_size):
                if (((y + sat_size) > h_limit) or
                        ((x + sat_size) > w_limit)):
                    break
                pred = queue.get()
                if pred is None:
                    break
                if pred.ndim == 3:
                    pred = pred.transpose((1, 2, 0))
                    canvas[y:y + map_size, x:x + map_size, :] += pred
                else:
                    canvas[y:y + map_size, x:x + map_size, 0] += pred
        print('offset:{} ({} sec)'.format(d, time.time() - st))

# get prediction image
def get_predict(gpu, sat_size, map_size, offset, channels, ortho, model, batchsize):

    xp = cuda.cupy if gpu >= 0 else np
    h_limit, w_limit = ortho.shape[0], ortho.shape[1]
    h_num = int(np.floor(h_limit / map_size))
    w_num = int(np.floor(w_limit / map_size))
    canvas_h = h_num * map_size - \
        (sat_size - map_size) + offset - 1
    canvas_w = w_num * map_size - \
        (sat_size - map_size) + offset - 1

    # to share 'canvas' between different threads
    canvas_ = Array(
        ctypes.c_float, canvas_h * canvas_w * channels)
    canvas = np.ctypeslib.as_array(canvas_.get_obj())
    canvas = canvas.reshape((canvas_h, canvas_w, channels))

    # prepare queues and threads
    patch_queue = Queue(maxsize=5)
    preds_queue = Queue()
    patch_worker = Process(
        target=create_minibatch, args=(sat_size, map_size, offset, h_limit, w_limit, batchsize, ortho, patch_queue))
    canvas_worker = Process(
        target=tile_patches, args=(sat_size, map_size, offset, h_limit, w_limit, canvas, preds_queue))
    patch_worker.start()
    canvas_worker.start()

    while True:
        minibatch = patch_queue.get()
        if minibatch is None:
            break
        with chainer.using_config('train', False):
            minibatch = Variable(xp.asarray(minibatch, dtype=xp.float32))
            preds = model(minibatch).data
        if gpu >= 0:
            preds = xp.asnumpy(preds)
        [preds_queue.put(pred) for pred in preds]

    preds_queue.put(None)
    patch_worker.join()
    canvas_worker.join()

    canvas = canvas[offset - 1:canvas_h - (offset - 1),
                    offset - 1:canvas_w - (offset - 1)]
    canvas /= offset

    return canvas

# main function
def predict(gpu, model, param, test_sat_dir, sat_size, map_size, channels, offset, batchsize):

    model_fn = os.path.basename(model)
    model = imp.load_source(model_fn.split('.')[0], model).model
    serializers.load_npz(param, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    epoch = re.search('epoch-([0-9]+)', param).groups()[0]
    if offset > 1:
        out_dir = '{}/ma_prediction_{}'.format(
            os.path.dirname(param), epoch)
    else:
        out_dir = '{}/prediction_{}'.format(os.path.dirname(param), epoch)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for fn in glob.glob('{}/*.tif*'.format(test_sat_dir)):

        img = cv.imread(fn)
        pred = get_predict(gpu, sat_size, map_size, offset, channels, img, model, batchsize)

        out_fn = '{}/{}.png'.format(
            out_dir, os.path.splitext(os.path.basename(fn))[0])
        cv.imwrite(out_fn, pred * 255)

        out_fn = '{}/{}.npy'.format(
            out_dir, os.path.splitext(os.path.basename(fn))[0])
        np.save(out_fn, pred)
