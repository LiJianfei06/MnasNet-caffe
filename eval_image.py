# -*- coding: UTF-8 -*-
"""
Created on Wed Aug 15 22:31:23 2018

python eval_image.py --proto deploy_MnasNet.prototxt --model ./model_save/MnasNet_model_cat_dog_iter_64000.caffemodel  --image ./cat.jpg

@author: lijianfei
"""
from __future__ import print_function
import argparse
import numpy as np
import sys
sys.path.append("/home/lijianfei/caffe-master-ljf/python")
sys.path.append("/home/lijianfei/caffe-master-ljf/python/caffe")
import caffe
import time


caffe.set_device(0)
caffe.set_mode_gpu()

    


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


def eval():
    nh, nw = 224, 224
    #img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)


    net = caffe.Net(args.proto, args.model, caffe.TEST)

    im = caffe.io.load_image(args.image)
    h, w, _ = im.shape
    if h < w:
        off = (w - h) / 2
        im = im[:, off:off + h]
    else:
        off = (h - w) / 2
        im = im[off:off + h, :]
    im = caffe.io.resize_image(im, [nh, nw])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    #transformer.set_mean('data', img_mean)
    transformer.set_input_scale('data', 0.00390625)

    net.blobs['data'].reshape(1, 3, nh, nw)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    for i in range(10):
        start = time.clock()
        out = net.forward()
        elapsed = (time.clock() - start)
        print("Time used:",elapsed," s")
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)

    label_names = np.loadtxt('labels.txt', str, delimiter='\t')
    for i in range(2):
        label = idx[i]
        print('%.2f - %s' % (prob[label], label_names[label]))
    return


if __name__ == '__main__':
    eval()
