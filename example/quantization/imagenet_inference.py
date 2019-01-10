# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
This script is used for calculating the accuracy of FP32 models or quantized models on the validation dataset
which was downloaded for calibration in imagenet_gen_qsym.py.
"""
import argparse
import logging
import os
import time
import mxnet as mx
from mxnet import nd
# from mxnet.contrib.quantization import *


def download_dataset(dataset_url, dataset_dir, logger_operator=None):
    if logger_operator is not None:
        logger_operator.info('Downloading dataset for inference from %s to %s' % (dataset_url, dataset_dir))
    mx.test_utils.download(dataset_url, dataset_dir)


def load_model(symbol_file_name, param_file_name, logger_operator=None):
    """
    load pre-trained model
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file_name)
    if logger_operator is not None:
        logger_operator.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file_name)
    if logger_operator is not None:
        logger_operator.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    args_params = {}
    auxs_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            args_params[name] = v
        if tp == 'aux':
            auxs_params[name] = v
    return symbol, args_params, auxs_params


def advance_data_iter(data_iter, n):
    """
    Get iterator of dataset
    """
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False


def score(symbol, args_params, auxs_params, data_set, devs, label_name_list, max_num_examples, logger_operator=None):
    """
    Get the score of inference
    """
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module(symbol=symbol, context=devs, label_names=[label_name_list, ])
    mod.bind(for_training=False,
             data_shapes=data_set.provide_data,
             label_shapes=data_set.provide_label)
    mod.set_params(args_params, auxs_params)

    tic = time.time()
    num = 0
    for batch in data_set:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    num_speed = num / (time.time() - tic)

    if logger_operator is not None:
        logger_operator.info('Finished inference with %d images' % num)
        logger_operator.info('Finished with %f images per second', num_speed)
        logger_operator.warn('Note: GPU performance is expected to be slower than CPU. '
                             'Please refer quantization/README.md for details')
        for m in metrics:
            logger_operator.info(m.get())


def benchmark_score(symbol_file_name, context, num_batch_size, num_batches, logger_operator=None):
    """
    Evaluate the model performance with benchmark
    """
    # get mod
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file_name)
    if logger_operator is not None:
        logger_operator.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)
    mod = mx.mod.Module(symbol=symbol, context=context)
    mod.bind(for_training=False,
             inputs_need_grad=False,
             data_shapes=[('data', (num_batch_size,)+data_shape)])
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data_set = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=context) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data_set, [])  # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches * num_batch_size / (time.time() - tic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--benchmark', type=bool, default=False, help='dummy data benchmark')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, required=False, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?'
                             'highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?'
                             'highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')

    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference', batch_size)

    rgb_mean = args.rgb_mean
    logger.info('rgb_mean = %s', rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    rgb_std = args.rgb_std
    logger.info('rgb_std = %s', rgb_std)
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}

    label_name = args.label_name
    logger.info('label_name = %s', label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s', str(data_shape))

    if args.benchmark is False:
        dataset = args.dataset
        download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset)
        logger.info('Dataset for inference: %s', dataset)

        # creating data iterator
        data = mx.io.ImageRecordIter(path_imgrec=dataset,
                                     label_width=1,
                                     preprocess_threads=data_nthreads,
                                     batch_size=batch_size,
                                     data_shape=data_shape,
                                     label_name=label_name,
                                     rand_crop=False,
                                     rand_mirror=False,
                                     shuffle=True,
                                     shuffle_chunk_seed=3982304,
                                     seed=48564309,
                                     **mean_args,
                                     **std_args)

        # loading model
        sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)

        # make sure that fp32 inference works on the same images as calibrated quantized model
        logger.info('Skipping the first %d batches', args.num_skipped_batches)
        data = advance_data_iter(data, args.num_skipped_batches)

        num_inference_images = args.num_inference_batches * batch_size
        logger.info('Running model %s for inference', symbol_file)
        score(sym, arg_params, aux_params, data, [ctx], label_name,
              max_num_examples=num_inference_images, logger_operator=logger)
    else:
        logger.info('Running model %s for inference', symbol_file)
        speed = benchmark_score(symbol_file, ctx, batch_size, args.num_inference_batches, logger)
        logger.info('batch size %2d, image/sec: %f', batch_size, speed)
