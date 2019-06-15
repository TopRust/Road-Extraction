
import numpy as np
import cv2 as cv

import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.serializers import hdf5
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from script.dataset import get_road
from script.transform import Road_Transform
import os
import imp

from chainer.datasets import SubDataset
from chainer.datasets import TransformDataset
from chainer.dataset import concat_examples

# learning rate * 0.1
def lr_drop(trainer):
    trainer.updater.get_optimizer('main').lr *= 0.1

# main function 
def train(n_process, modelpy, gpu_id, fliplr, rotate, norm, image_side, label_side, out_directory, epoch):

    # get train valid dataset
    train, valid = get_road()

    # set transform function
    Trans = Road_Transform(fliplr, rotate, norm, image_side, image_side, label_side, label_side)
    
    # transform dataset 
    train = TransformDataset(train, Trans.transform)
    valid = TransformDataset(valid, Trans.transform)

    batchsize = 128

    # multiprocess
    train_iter = iterators.MultiprocessIterator(train, batchsize, True, True, n_process, 1)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, False, False, n_process, 1)
    
    # init network
    model_fn = os.path.basename(modelpy)
    model = imp.load_source(model_fn.split('.')[0], modelpy).model

    # use gpu
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epoch = 400

    # Wrap your model by Classifier and include the process of loss calculation within your model.
    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
    model = L.Classifier(model)

    # selection of your optimizing method
    optimizer = optimizers.MomentumSGD(lr=0.005, momentum=0.9)

    # Give the optimizer a reference to the model
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # Get an updater that uses the Iterator and Optimizer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Setup a Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=out_directory)

    from chainer.training import extensions

    trainer.extend(extensions.LogReport()) # generate report
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}')) # save updater
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}')) # save model
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id)) # validation

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time'])) # show loss and accuracy
    trainer.extend(extensions.ProgressBar()) # show trainning progress
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png')) # loss curve
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png')) # accuracy curve
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(lr_drop, trigger=(100, 'epoch'))# learning rate * 0.1 per 100 epoch

    # load trainer, resume trainning 
    if epoch != 0:
        serializers.load_npz('{}/snapshot_epoch-{}'.format(out_directory, epoch), trainer)
    trainer.run()

