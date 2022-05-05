"""7. Fine-tuning SOTA video models on your own dataset
=======================================================

This is a video action recognition tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Fine-tuning is an important way to obtain good video models on your own data when you don't have large annotated dataset or don't have the
computing resources to train a model from scratch for your use case.
In this tutorial, we provide a simple unified solution.
The only thing you need to prepare is a text file containing the information of your videos (e.g., the path to your videos),
we will take care of the rest.
You can start fine-tuning from many popular pre-trained models (e.g., I3D, I3D-nonlocal, SlowFast) using a single command line.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. note::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train_recognizer.py<../../../scripts/action-recognition/train_recognizer.py>`

    For more training command options, please run ``python train_recognizer.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ for training commands of reproducing the pretrained model.


First, let's import the necessary libraries into python.

"""
from __future__ import division

import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import mxnet.ndarray as nd

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.model_zoo import i3d_resnet50_v1_ucf101
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory
from gluoncv.utils import export_block
from gluoncv.utils.filesystem import try_import_decord

import pandas as pd
import torch
import torchvision.models as models


######################################################################
# Custom DataLoader
# ------------------
#
# We provide a general dataloader for you to use on your own dataset. Your data can be stored in any hierarchy,
# can be stored in either video format or already decoded to frames. The only thing you need
# to prepare is a text file, ``train.txt``.
#
# If your data is stored in image format (already decoded to frames). Your ``train.txt`` should look like:
#
# ::
#
#     video_001 200 0
#     video_001 200 0
#     video_002 300 0
#     video_003 100 1
#     video_004 400 2
#     ......
#     video_100 200 10
#
# There are three items in each line, separated by spaces.
# The first item is the path to your training videos, e.g., video_001.
# It should be a folder containing the frames of video_001.mp4.
# The second item is the number of frames in each video, e.g., 200.
# The third item is the label of the videos, e.g., 0.
#
# If your data is stored in video format. Your ``train.txt`` should look like:
#
# ::
#
#     video_001.mp4 200 0
#     video_001.mp4 200 0
#     video_002.mp4 300 0
#     video_003.mp4 100 1
#     video_004.mp4 400 2
#     ......
#     video_100.mp4 200 10
#
# Similarly, there are three items in each line, separated by spaces.
# The first item is the path to your training videos, e.g., video_001.mp4.
# The second item is the number of frames in each video. But you can put any number here
# because our video loader will compute the number of frames again automatically during training.
# The third item is the label of that video, e.g., 0.
#
#
# Once you prepare the ``train.txt``, you are good to go.
# Just use our general dataloader `VideoClsCustom <https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/kinetics400/classification.py>`_ to load your data.
#
# In this tutorial, we will use UCF101 dataset as an example.
# For your own dataset, you can just replace the value of ``root`` and ``setting`` to your data directory and your prepared text file.
# Let's first define some basics.

class OPT():
    def __init__(self, data_path, model_path, save_path) -> None:
        self.data_dir = data_path
        self.need_root = None
        self.data_list = None
        self.dtype = np.float32
        self.gpu_id = 0
        self.mode = 'hybrid'
        self.use_pretrained = True
        self.resume_params = model_path
        self.num_segments = 1
        self.new_height = 224
        self.new_width = 224
        self.new_length = 60
        self.new_step = 1
        self.num_classes = 13
        self.video_loader = True
        self.use_decod = True
        self.num_crop = 1
        self.save_dir = save_path
        
        self.slowfast = False
        self.input_size = 224
        


num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
per_device_batch_size = 2
num_workers = 0
batch_size = per_device_batch_size * num_gpus

def main(data_path, model_path, save_path, mask_id):
    f = open(data_path, 'r')
    data_list = f.readlines()
    print('Load %d video samples.' % len(data_list))

    opt = OPT(data_path, model_path, save_path)

    video_utils = VideoClsCustom(root=os.path.expanduser(r'.\\'),
                                setting=data_list,
                                num_segments=opt.num_segments,
                                num_crop=opt.num_crop,
                                new_length=opt.new_length,
                                new_step=opt.new_step,
                                video_loader=opt.video_loader,
                                use_decord=opt.use_decod,
                                lazy_init=True
                                )
    # print('Load %d training samples.' % len(train_dataset))
    # train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
    #                                 shuffle=False, num_workers=num_workers)


    ################################################################
    # Custom Network
    # --------------
    #
    # You can always define your own network architecture. Here, we want to show how to fine-tune on a pre-trained model.
    # Since I3D model is a very popular network, we will use I3D with ResNet50 backbone trained on Kinetics400 dataset (i.e., ``i3d_resnet50_v1_kinetics400``) as an example.
    #
    # For simple fine-tuning, people usually just replace the last classification (dense) layer to the number of classes in your dataset
    # without changing other things. In GluonCV, you can get your customized model with one line of code.
    
    #ref: 
    # https://github.com/dmlc/gluon-cv/issues/229#issuecomment-446468952
    # https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/getting-started/crash-course/5-predict.html
    # https://mxnet.apache.org/versions/1.9.0/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html
    # net = get_model(name= model_name, nclass=13, pretrained_base=True)
    # net.load_parameters(model_path, ctx=mx.gpu(0))
    
    net = i3d_resnet50_v1_ucf101(nclass=13, pretrained=False, pretrained_base=False,
                                 ctx=mx.gpu(0))
    net.load_parameters(opt.resume_params)
    print(net)


    
    net.hybridize(static_alloc=True, static_shape=True)
    
    classes = ["walk", "sit_down", "sit_up",
                "climb_down", "climb_up",
                "squat_down", "squat_up",
                "take_up", "throw",
                "tumble_down", "tumble_up",
                "pick_up", "pick_down"]
    
    # predict_record = pd.DataFrame(columns=['sample_id', 'true_label', 'predict_label'])
    predict_record = pd.DataFrame(columns=['sample_id', 'sub_id', 'cam_id', 'true_label', 'predict_label'])
    
    
    start_time = time.time()
    for vid, vline in enumerate(data_list):
        video_path = vline.split()[0]
        splited_path = os.path.normpath(video_path).split(os.path.sep)
        video_name = splited_path[-1]
        # print("current process: {}".format(video_path))
        # if opt.need_root:
        #     video_path = os.path.join(opt.data_dir, video_path)
        video_data = read_data(opt, video_path, transform_train, video_utils)
        video_input = video_data.as_in_context(mx.gpu(0))
        pred = net(video_input.astype(opt.dtype, copy=False))
        
        # save predict results
        pred_label = np.argmax(pred.asnumpy())
        
        # if classes:
        #     pred_label = classes[pred_label]
        result_arr = [vid, splited_path[-3], splited_path[-2],  int(vline.split()[-1]), pred_label]
        
        predict_record.loc[len(predict_record)] = result_arr
    
    predict_record.to_csv(os.path.join(opt.save_dir, "predict_results_{:s}.csv".format(mask_id)), header=True, index=False)
    
    end_time = time.time()
    print('Total inference time is %4.2f minutes' % ((end_time - start_time) / 60))
    

def read_data(opt, video_name, transform, video_utils):

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=opt.new_width, height=opt.new_height)
    duration = len(decord_vr)

    opt.skip_length = opt.new_length * opt.new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if opt.video_loader:
        if opt.slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if opt.slowfast:
        sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt.new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    return nd.array(clip_input)

def verify_loaded_model(net):
    """Run inference using ten random images.
    Print both input and output of the model"""

    def transform(data, label):
        return data.astype(np.float32)/255, label.astype(np.float32)

    # Load ten random images from the test dataset
    sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                  10, shuffle=True)

    for data, label in sample_data:

        # Display the predictions
        data = nd.transpose(data, (0, 3, 1, 2))
        out = net(data.as_in_context(ctx))
        predictions = nd.argmax(out, axis=1)
        print('Model predictions: ', predictions.asnumpy())

        break


######################################################################
# We can see that the training accuracy increase quickly.
# Actually, if you look back tutorial 4 (Dive Deep into Training I3D mdoels on Kinetcis400) and compare the training curve,
# you will see fine-tuning can achieve much better result using much less time.
# Try fine-tuning other SOTA video models on your own dataset and see how it goes.

def get_cover_type():
    cover_type = ['bottom', 'middle', 'top']
    return cover_type

def get_cover_rate():
    return [0.3, 0.5, 0.7]

# list whole pattern of created masks
# return: mask_state[mask_rate, mask_type]
def get_mask_state():
    cover_type = get_cover_type()
    cover_rate = get_cover_rate()
    mask_state = []

    for c_rate in cover_rate:
        for c_type in cover_type:
            mask_state.append(c_type + str(c_rate))

    return np.array(mask_state).reshape(len(cover_rate), len(cover_type))

def get_mask_class():
    # mask_class = ["grid_stochastic_mask", "stochastic_position_mask", "grid_vertical_mask", "grid_horizontal_mask"]
    mask_class = ["grid_stochastic_mask", "stochastic_position_mask", "grid_horizontal_mask"]
    return np.array(mask_class)

def get_total_mask_status():
    total = np.append(get_mask_state().ravel(), get_mask_class())
    return total

def DF_to_CSV(df, csv_dirpath, csv_name, index =False):
    # check params are OK to use
    if csv_name.split('.')[-1] != 'csv':
        csv_name += '.csv'
        
    if not os.path.exists(csv_dirpath):
        print("lib_IO_fun.py-> DF_to_CSV: Path is not exists, create one:%s"%csv_dirpath)
        os.makedirs(csv_dirpath)
        
    df.to_csv(os.path.join(csv_dirpath, csv_name), header=True, index =index)
    
    return os.path.join(csv_dirpath, csv_name)

if __name__ == "__main__":
    model_name = "i3d_resnet50_v1_kinetics400"
    model_rootpath = r"K:\ActionRecognition_OpenPose\Comp_i3d_resnet50_v1_kinetics400_202203151419"
    model_subpath = r"model\checkpoint\ckpt_epoch_100.params"
    model_path = os.path.join(model_rootpath, model_subpath)
    
    # data_rootpath = r"K:\ActionRecognition_data\18_mask_data\video_data"
    data_rootpath = r"J:\ActionRecognition_data\18_mask_data\video_data"
    
    target_name = "estimation_{:s}_{}".format(model_name, time.strftime("%Y%m%d%H%M", time.localtime()))
    
    for mask_id in get_total_mask_status():
        if mask_id == "bottom0.3":
            continue
        data_path = 'label_record_{}.txt'.format(mask_id)
        data_path = os.path.join(data_rootpath, data_path)
    
        save_path = os.path.join(model_rootpath, target_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        main(data_path, model_path, save_path, mask_id)

