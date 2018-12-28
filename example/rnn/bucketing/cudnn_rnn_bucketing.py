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
Generate the Sherlock Holmes language model by using RNN
"""
import argparse
import mxnet as mx

parser = argparse.ArgumentParser(description="Train RNN on Sherlock Holmes",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=200,
                    help='embedding layer size')
parser.add_argument('--bidirectional', action='store_true',
                    help='uses bidirectional layers if specified')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
# When training a deep, complex model *on multiple GPUs* it's recommended to
# stack fused RNN cells (one layer per cell) together instead of one with all
# layers. The reason is that fused RNN cells don't set gradients to be ready
# until the computation for the entire layer is completed. Breaking a
# multi-layer fused RNN cell into several one-layer ones allows gradients to be
# processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.0',
                    help='dropout probability (1.0 - keep probability)')
parser.add_argument('--rnntype', type=str, default='lstm',
                    help='rnn type: gru, lstm, rnn_tanh and rnn_relu are supported')

# buckets = [32]
buckets = [10, 20, 30, 40, 50, 60]

start_label = 1
invalid_label = 0


def tokenize_text(file_name, vocab=None, invalid_label_list=-1, start_label_list=0):
    lines = open(file_name).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label_list,
                                               start_label=start_label_list)
    return sentences, vocab


def get_data(layout):
    """
    Load data for RNN training and validation
    """
    train_sent, vocab = tokenize_text("./data/sherlockholmes.train.txt", start_label_list=start_label,
                                      invalid_label_list=invalid_label)
    val_sent, _ = tokenize_text("./data/sherlockholmes.test.txt", vocab=vocab, start_label_list=start_label,
                                invalid_label_list=invalid_label)

    data_train = mx.rnn.BucketSentenceIter(train_sent, args.batch_size, buckets=buckets,
                                           invalid_label=invalid_label, layout=layout)
    data_val = mx.rnn.BucketSentenceIter(val_sent, args.batch_size, buckets=buckets,
                                         invalid_label=invalid_label, layout=layout)
    return data_train, data_val, vocab


def train(args_list):
    """
    Process RNN training of the Sherlock Holmes language model
    """
    data_train, data_val, vocab = get_data('TN')
    if args_list.stack_rnn:
        cell = mx.rnn.SequentialRNNCell()
        for i in range(args_list.num_layers):
            cell.add(mx.rnn.FusedRNNCell(args_list.num_hidden, num_layers=1,
                                         mode=args_list.rnntype, prefix='%s_l%d' % (args_list.rnntype, i),
                                         bidirectional=args_list.bidirectional))
            if args_list.dropout > 0 and i < args_list.num_layers - 1 and args_list.rnntype == 'lstm':
                cell.add(mx.rnn.DropoutCell(args_list.dropout, prefix='%s_d%d' % (args_list.rnntype, i)))
    else:
        cell = mx.rnn.FusedRNNCell(args_list.num_hidden, num_layers=args_list.num_layers, dropout=args_list.dropout,
                                   mode=args_list.rnntype, bidirectional=args_list.bidirectional)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args_list.num_embed, name='embed')

        output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')

        pred = mx.sym.Reshape(output, shape=(-1, args_list.num_hidden*(1+args_list.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args_list.gpus:
        contexts = [mx.gpu(int(i)) for i in args_list.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(sym_gen=sym_gen,
                                   default_bucket_key=data_train.default_bucket_key,
                                   context=contexts)

    if args_list.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(cell, args_list.model_prefix, args_list.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {'learning_rate': args_list.lr, 'wd': args_list.wd}

    if args_list.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args_list.mom

    model.fit(train_data=data_train,
              eval_data=data_val,
              eval_metric=mx.metric.Perplexity(invalid_label),
              kvstore=args_list.kv_store,
              optimizer=args_list.optimizer,
              optimizer_params=opt_params,
              initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
              arg_params=arg_params,
              aux_params=aux_params,
              begin_epoch=args_list.load_epoch,
              num_epoch=args_list.num_epochs,
              batch_end_callback=mx.callback.Speedometer(args_list.batch_size, args_list.disp_batches,
                                                         auto_reset=False),
              epoch_end_callback=mx.rnn.do_rnn_checkpoint(cell, args_list.model_prefix, 1)
              if args_list.model_prefix else None)


def test(args_list):
    """
    Generate the testing for RNN
    """
    assert args_list.model_prefix, "Must specifiy path to load from"
    _, data_val, vocab = get_data('NT')

    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                                    mode=args.rnntype, bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            if args.rnntype == 'lstm':
                cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='%s_%dl0_'%(args.rnntype, i))
                if args.bidirectional:
                    cell = mx.rnn.BidirectionalCell(cell, mx.rnn.LSTMCell(num_hidden=args.num_hidden,
                                                                          prefix='%s_%dr0_' % (args.rnntype, i)),
                                                    output_prefix='bi_%s_%d' % (args.rnntype, i))
            elif args.rnntype == 'gru':
                cell = mx.rnn.GRUCell(num_hidden=args.num_hidden, prefix='%s_%dl0_' % (args.rnntype, i))
                if args.bidirectional:
                    cell = mx.rnn.BidirectionalCell(cell, mx.rnn.GRUCell(num_hidden=args.num_hidden,
                                                                         prefix='%s_%dr0_' % (args.rnntype, i)),
                                                    output_prefix='bi_%s_%d' % (args.rnntype, i))
            elif args.rnntype == 'rnn_tanh':
                cell = mx.rnn.RNNCell(num_hidden=args.num_hidden, activation='tanh',
                                      prefix='%s_%dl0_' % (args.rnntype, i))
                if args.bidirectional:
                    cell = mx.rnn.BidirectionalCell(cell, mx.rnn.RNNCell(num_hidden=args.num_hidden, activation='tanh',
                                                                         prefix='%s_%dr0_' % (args.rnntype, i)),
                                                    output_prefix='bi_%s_%d' % (args.rnntype, i))
            elif args.rnntype == 'rnn_relu':
                cell = mx.rnn.RNNCell(num_hidden=args.num_hidden, activation='relu',
                                      prefix='%s_%dl0_' % (args.rnntype, i))
                if args.bidirectional:
                    cell = mx.rnn.BidirectionalCell(cell, mx.rnn.RNNCell(num_hidden=args.num_hidden,
                                                                         activation='relu',
                                                                         prefix='%s_%dr0_' % (args.rnntype, i)),
                                                    output_prefix='bi_%s_%d' % (args.rnntype, i))

            stack.add(cell)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args.num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(sym_gen=sym_gen,
                                   default_bucket_key=data_val.default_bucket_key,
                                   context=contexts)
    model.bind(data_val.provide_data, data_val.provide_label, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params)

    model.score(data_val, mx.metric.Perplexity(invalid_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)
