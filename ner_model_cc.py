# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding
from loader import load_sentences
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from data_utils import data_iterator, load_word2vec
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

import time
import numpy as np
import tensorflow as tf
import sys
import os
import codecs
import pickle
import itertools


pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pkg_path)

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "ner_data")
train_dir = os.path.join(file_path, "ner_model_cc_debug")

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("seg_data_path", data_path, "data_path")
flags.DEFINE_string("seg_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("seg_scope_name", "seg_var_scope", "Define SEG Variable Scope Name")
flags.DEFINE_string("vector_file", os.path.join(data_path, "wiki_100.utf8"), "word vectors file")
flags.DEFINE_boolean("stack", False, "use a second LSTM layer")
flags.DEFINE_integer("max_epoch", 20, "max epochs")
flags.DEFINE_integer("vocab_size", 16116, "vocab size")
flags.DEFINE_integer("target_num", 13, "target nums")
flags.DEFINE_integer("embedding_size", 100, "char embedding size")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("seg_embedding_size", 20, "seg embedding size")
flags.DEFINE_integer("seg_nums", 4, "seg nums")
flags.DEFINE_integer("hidden_size", 100, "hidden_size")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("keep_prob", 0.5, "drop out keep prob")
flags.DEFINE_string("train_file",   os.path.join(data_path, "train_data.txt"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(data_path, "dev_data.txt"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(data_path, "test_data.txt"),   "Path for test data")
flags.DEFINE_string("test_result_file", os.path.join(data_path, "ner_model_cc_result.txt"), "Path for result")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")

FLAGS = flags.FLAGS


def data_type():
    return tf.float32


class Segmenter(object):
    """The Segmenter Model."""

    def __init__(self, config, init_embedding=None):
        self.batch_size = batch_size = config.batch_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.seg_embedding_size = config.seg_embedding_size
        self.seg_nums = config.seg_nums

        # Define input and target tensors
        self._input_data = tf.placeholder(tf.int32, [batch_size, None], name='input_data')
        self._targets = tf.placeholder(tf.int32, [batch_size, None], name='targets')
        self._seq_len = tf.placeholder(tf.int32, [batch_size], name='seq_len')
        self._seg_data = tf.placeholder(tf.int32, [batch_size, None], name='seg_data')
        self._max_seq_len = tf.placeholder(tf.int32, shape=[1], name='max_seq_len')

        with tf.device("/cpu:0"):
            if init_embedding is None:
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type(), trainable=True)
            else:
                self.embedding = tf.Variable(init_embedding, name="embedding", dtype=data_type(), trainable=True)
            # embedding for seg data
            self.seg_embedding = tf.get_variable(name="seg_embedding", shape=[self.seg_nums, self.seg_embedding_size], dtype=data_type(), trainable=True)
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        seg_inputs = tf.nn.embedding_lookup(self.seg_embedding, self._seg_data)
        inputs = tf.concat([inputs, seg_inputs], axis=-1)

        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[1], name="Dropout")
        self._tt = tf.reshape(self._dropout_keep_prob, shape=[])
        inputs = tf.nn.dropout(inputs, self._tt)

        # 字窗口特征和分词特征
        inputs = tf.reshape(inputs, [batch_size, -1, self.embedding_size + self.seg_embedding_size])
        self._lstm_inputs = inputs
        seg_data_reshape = tf.reshape(self._seg_data, [batch_size, -1, 1])
        seg_data_reshape = tf.cast(seg_data_reshape, dtype=data_type())
        self._loss, self._logits, self._trans, self._seq_len_plus1 = _bilstm_model(inputs, self._targets, self._seq_len, config, seg_data_reshape, self._max_seq_len)
        # self._viterbi_sequence_len = tf.placeholder(dtype=tf.int32,  name='viterbi_len')
        # self._viterbi_logits = tf.placeholder(dtype=data_type(), shape=[None, self.seg_nums+1], name='viterbi_logits')

        self._viterbi_sequence, _ = crf_decode(self._logits, self._trans, self._seq_len+1)
        print("_viterbi_sequence name: ", self._viterbi_sequence.name, type(self._viterbi_sequence))

        # print("input data name: ",self._input_data.name)
        # print("targets name: ",self._targets.name)
        # print("seq len name: ",self._seq_len.name)
        # print("seg data name: ",self._seg_data.name)
        # print("max seq len name: ", self._max_seq_len.name)
        # print(self._loss.name)
        # print(self._logits.name)
        # print(self._trans.name)
        with tf.variable_scope("train_ops") as scope:
            # Gradients and SGD update operation for training the model.
            self._lr = tf.Variable(0.001, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
            self.optimizer = tf.train.AdamOptimizer(self._lr)
            self._train_op = self.optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            # self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
            # self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr, feed_dict={self._lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def seg_data(self):
        return self._seg_data

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def loss(self):
        return self._loss

    @property
    def logits(self):
        return self._logits

    @property
    def trans(self):
        return self._trans

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def lstm_inputs(self):
        return self._lstm_inputs

    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def seq_len_plus1(self):
        return self._seq_len_plus1

    @property
    def viterbi_sequence(self):
        return self._viterbi_sequence


# seg model configuration, set target num, and input vocab_size
class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.05
    learning_rate = FLAGS.lr
    max_grad_norm = 5
    hidden_size = FLAGS.hidden_size
    embedding_size = FLAGS.embedding_size
    max_epoch = 5
    max_max_epoch = FLAGS.max_epoch
    stack = FLAGS.stack
    keep_prob = FLAGS.keep_prob  # There is one dropout layer on input tensor also, don't set lower than 0.9
    lr_decay = 1 / 1.15
    batch_size = FLAGS.batch_size  # single sample batch
    vocab_size = FLAGS.vocab_size
    target_num = FLAGS.target_num  # SEG tagging tag number for ChineseNER
    bi_direction = True  # LSTM or BiLSTM
    seg_embedding_size = FLAGS.seg_embedding_size
    seg_nums = FLAGS.seg_nums


def get_config():
    return LargeConfigChinese()


def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)


def _bilstm_model(inputs, targets, seq_len, config, seg_data, max_seq_len):
    '''
    @Use BasicLSTMCell, MultiRNNCell method to build LSTM model
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num  # target output number
    seq_len = tf.cast(seq_len, tf.int32)

    fw_cell = lstm_cell(hidden_size)
    bw_cell = lstm_cell(hidden_size)

    with tf.variable_scope("seg_bilstm"):
        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_len,
            scope='layer_1'
        )
        output = tf.concat(axis=2, values=[forward_output, backward_output])
        # if stack 又添加一层双向LSTM
        if config.stack:
            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                output,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_2'
            )
            output = tf.concat(axis=2, values=[forward_output, backward_output])

            # outputs is a length T list of output vectors, which is [batch_size*maxlen, 2 * hidden_size]

        # 直接在CRF层输入加入分词特征
        output = tf.concat(values=[output, seg_data], axis=-1)
        output = tf.reshape(output, [-1, 2 * hidden_size+1])

        # 加一个tanh激活函数
        # W = tf.get_variable("W", shape=[hidden_size * 2, hidden_size], dtype=data_type())
        # b = tf.get_variable("b", shape=[hidden_size], dtype=data_type(), initializer=tf.zeros_initializer())
        # hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))#hidden 的size:[-1,hidden_size]

        softmax_w = tf.get_variable("softmax_w", [2*hidden_size+1, target_num], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits, [batch_size, -1, target_num])
    # CRF层
    with tf.variable_scope("loss") as scope:
        # CRF log likelihood
        small = -1000.0
        # pad logits for crf loss
        start_logits = tf.concat([small * tf.ones(shape=[batch_size, 1, target_num]), tf.zeros(shape=[batch_size, 1, 1])], axis=-1)
        max_seq_len_tt = tf.reshape(max_seq_len, shape=[])
        pad_logits = tf.cast(small * tf.ones([batch_size, max_seq_len_tt, 1]), tf.float32)
        new_logits = tf.concat([logits, pad_logits], axis=-1)
        new_logits = tf.concat([start_logits, new_logits], axis=1, name='crf_logits')
        new_targets = tf.concat([tf.cast(target_num * tf.ones([batch_size, 1]), tf.int32), targets], axis=-1)
        # CRF encode
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(new_logits, new_targets, seq_len+1)
        loss = tf.reduce_mean(-log_likelihood, name='crf_losses')

        # CRF decode
        # crf_decode(new_logits, transition_params, seq_len+1)
        # for decode_vitrebi, decode_len in zip()
        # new_logits_array = new_logits[1]
    return loss, new_logits, transition_params, seq_len+1


def _crf_decode(logits, trans, seq_len, batch_size):
    decode_results = []


    # for logits_, l_ in zip(logits_array, seq_len_array):
    for i in range(0, batch_size):
        l_ = seq_len[i]
        logits_ = logits[i][:l_ + 1]
        print("decode logits shape: ", logits_.shape)
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)
        decode_results.append(viterbi_sequence[1:])
    return decode_results


def run_epoch(session, model, data, eval_op, batch_size, verbose=False):
    """Runs the model on the given data."""
    losses = 0.0
    iters = 0.0

    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, batch_size)

    for x, y, l, seg in zip(xArray, yArray, lArray, segArray):
        fetches = [model.loss, model.logits, eval_op]
        feed_dict = dict()
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = [FLAGS.keep_prob]
        feed_dict[model.max_seq_len] = [max(l)]
        loss, logits, _ = session.run(fetches, feed_dict)
        losses += loss
        iters += 1

        if verbose and iters % 50 == 0:
            print("percent: %.3f ,mean losses: %.3f" % (iters / float(len(xArray)), losses / iters / len(xArray)))

    return losses / iters / len(xArray)


def ner_evaluate(session, model, data, eval_op, batch_size, tag_to_id):
    correct_labels = 0
    total_labels = 0

    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, batch_size)

    per_yp_wordnum = 0
    per_yt_wordnum = 0
    per_cor_num = 0
    org_yp_wordnum = 0
    org_yt_wordnum = 0
    org_cor_num = 0
    loc_yp_wordnum = 0
    loc_yt_wordnum = 0
    loc_cor_num = 0

    for x, y, l, seg in zip(xArray, yArray, lArray, segArray):
        fetches = [model.loss, model.logits, model.trans]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = [1.0]#evaluate的时候keep设置为1
        feed_dict[model.max_seq_len] = [max(l)]
        loss, logits, trans = session.run(fetches, feed_dict)
        for logits_, y_, l_ in zip(logits, y, l):
            logits_ = logits_[:l_+1]
            y_ = y_[:l_]

            # crf decode
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)
            viterbi_sequence = viterbi_sequence[1:]

            # 计算评测
            B_PER_id = tag_to_id["B-PER"]
            E_PER_id = tag_to_id["E-PER"]
            S_PER_id = tag_to_id["S-PER"]
            B_LOC_id = tag_to_id["B-LOC"]
            E_LOC_id = tag_to_id["E-LOC"]
            S_LOC_id = tag_to_id["S-LOC"]
            B_ORG_id = tag_to_id["B-ORG"]
            E_ORG_id = tag_to_id["E-ORG"]
            S_ORG_id = tag_to_id["S-ORG"]

            per_yp_wordnum += viterbi_sequence.count(E_PER_id) + viterbi_sequence.count(S_PER_id)
            per_yt_wordnum += y_.count(S_PER_id) + y_.count(E_PER_id)
            org_yp_wordnum += viterbi_sequence.count(E_ORG_id) + viterbi_sequence.count(S_ORG_id)
            org_yt_wordnum += y_.count(E_ORG_id) + y_.count(S_ORG_id)
            loc_yp_wordnum += viterbi_sequence.count(E_LOC_id) + viterbi_sequence.count(S_LOC_id)
            loc_yt_wordnum += y_.count(E_LOC_id) + y_.count(S_LOC_id)
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += l_

            for i in range(0, len(y_)):
                # 计算PER
                if y_[i] == S_PER_id:
                    if y_[i] == viterbi_sequence[i]:
                        per_cor_num += 1
                elif y_[i] == B_PER_id:
                    flag = True
                    while True:
                        if y_[i] != viterbi_sequence[i]:
                            flag = False
                        if y_[i] == E_PER_id:
                            break
                        i += 1
                    if flag:
                        per_cor_num += 1
                # 计算ORG
                elif y_[i] == S_ORG_id:
                    if y_[i] == viterbi_sequence[i]:
                        org_cor_num += 1
                elif y_[i] == B_ORG_id:
                    flag = True
                    while True:
                        if y_[i] != viterbi_sequence[i]:
                            flag = False
                        if y_[i] == E_ORG_id:
                            break
                        i += 1
                    if flag:
                        org_cor_num += 1
                # 计算LOC
                elif y_[i] == S_LOC_id:
                    if y_[i] == viterbi_sequence[i]:
                        loc_cor_num += 1
                elif y_[i] == B_LOC_id:
                    flag = True
                    while True:
                        if y_[i] != viterbi_sequence[i]:
                            flag = False
                        if y_[i] == E_LOC_id:
                            break
                        i += 1
                    if flag:
                        loc_cor_num += 1
    per_P = per_cor_num/float(per_yp_wordnum)
    per_R = per_cor_num/float(per_yt_wordnum)
    per_F = 2 * per_P * per_R / (per_P + per_R)

    loc_P = loc_cor_num / float(loc_yp_wordnum)
    loc_R = loc_cor_num / float(loc_yt_wordnum)
    loc_F = 2 * loc_P * loc_R / (loc_P + loc_R)

    org_P = org_cor_num / float(org_yp_wordnum)
    org_R = org_cor_num / float(org_yt_wordnum)
    org_F = 2 * org_P * org_R / (org_P + org_R)

    accuracy = 100.0 * correct_labels / float(total_labels)
    total_P = (per_cor_num + loc_cor_num + org_cor_num)/float(per_yp_wordnum+loc_yp_wordnum+org_yp_wordnum)
    total_R = (per_cor_num + loc_cor_num + org_cor_num)/float(per_yt_wordnum+loc_yt_wordnum+org_yt_wordnum)
    total_F = 2*total_P*total_R /(total_P + total_R)

    return accuracy, total_P, total_R, total_F, per_P, per_R, per_F, loc_P, loc_R, loc_F, org_P, org_R, org_F


class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
  """Computes the forward decoding in a linear-chain CRF.
  """

  def __init__(self, transition_params):
    """Initialize the CrfDecodeForwardRnnCell.
    Args:
      transition_params: A [num_tags, num_tags] matrix of binary
        potentials. This matrix is expanded into a
        [1, num_tags, num_tags] in preparation for the broadcast
        summation occurring within the cell.
    """
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      scope: Unused variable scope of this cell.
    Returns:
      backpointers: [batch_size, num_tags], containing backpointers.
      new_state: [batch_size, num_tags], containing new score values.
    """
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    state = array_ops.expand_dims(state, 2)                         # [B, O, 1]

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension.
    # [B, O, 1] + [1, O, O] -> [B, O, O]
    transition_scores = state + self._transition_params             # [B, O, O]
    new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
    backpointers = math_ops.argmax(transition_scores, 1)
    backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)    # [B, O]
    return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
  """Computes backward decoding in a linear-chain CRF.
  """

  def __init__(self, num_tags):
    """Initialize the CrfDecodeBackwardRnnCell.
    Args:
      num_tags
    """
    self._num_tags = num_tags

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeBackwardRnnCell.
    Args:
      inputs: [batch_size, num_tags], backpointer of next step (in time order).
      state: [batch_size, 1], next position's tag index.
      scope: Unused variable scope of this cell.
    Returns:
      new_tags, new_tags: A pair of [batch_size, num_tags]
        tensors containing the new tag indices.
    """
    state = array_ops.squeeze(state, axis=[1])                # [B]
    batch_size = array_ops.shape(inputs)[0]
    b_indices = math_ops.range(batch_size)                    # [B]
    indices = array_ops.stack([b_indices, state], axis=1)     # [B, 2]
    new_tags = array_ops.expand_dims(
        gen_array_ops.gather_nd(inputs, indices),             # [B]
        axis=-1)                                              # [B, 1]

    return new_tags, new_tags


def crf_decode(potentials, transition_params, sequence_length):
  """Decode the highest scoring sequence of tags in TensorFlow.
  This is a function for tensor.
  Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor, matrix of
              unary potentials.
    transition_params: A [num_tags, num_tags] tensor, matrix of
              binary potentials.
    sequence_length: A [batch_size] tensor, containing sequence lengths.
  Returns:
    decode_tags: A [batch_size, max_seq_len] tensor, with dtype tf.int32.
                Contains the highest scoring tag indicies.
    best_score: A [batch_size] tensor, containing the score of decode_tags.
  """
  # For simplicity, in shape comments, denote:
  # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
  num_tags = potentials.get_shape()[2].value

  # Computes forward decoding. Get last score and backpointers.
  crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
  initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
  initial_state = array_ops.squeeze(initial_state, axis=[1])      # [B, O]
  inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])   # [B, T-1, O]
  backpointers, last_score = rnn.dynamic_rnn(
      crf_fwd_cell,
      inputs=inputs,
      sequence_length=sequence_length - 1,
      initial_state=initial_state,
      time_major=False,
      dtype=dtypes.int32)             # [B, T - 1, O], [B, O]
  backpointers = gen_array_ops.reverse_sequence(
      backpointers, sequence_length - 1, seq_dim=1)               # [B, T-1, O]

  # Computes backward decoding. Extract tag indices from backpointers.
  crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
  initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),
                                dtype=dtypes.int32)               # [B]
  initial_state = array_ops.expand_dims(initial_state, axis=-1)   # [B, 1]
  decode_tags, _ = rnn.dynamic_rnn(
      crf_bwd_cell,
      inputs=backpointers,
      sequence_length=sequence_length - 1,
      initial_state=initial_state,
      time_major=False,
      dtype=dtypes.int32)           # [B, T - 1, 1]
  decode_tags = array_ops.squeeze(decode_tags, axis=[2])           # [B, T - 1]
  decode_tags = array_ops.concat([initial_state, decode_tags], axis=1)  # [B, T]
  decode_tags = gen_array_ops.reverse_sequence(
      decode_tags, sequence_length, seq_dim=1)                     # [B, T]

  best_score = math_ops.reduce_max(last_score, axis=1)             # [B]
  return decode_tags, best_score


#test viterbi_decode
def test_viterbi_decode(score, transition_params):
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1)
        print("v shape:", v.shape)
        print("v: ", v)
        print("trellist t shape: ", trellis[t].shape)
        print(trellis[t])
        trellis[t] = score[t] + np.max(v, 0)
        print("max v:", np.max(v, 0))
        print("after: ", trellis[t])
        backpointers[t] = np.argmax(v, 0)
        print("back pointers: ", backpointers[t])
        break

    # viterbi = [np.argmax(trellis[-1])]
    # for bp in reversed(backpointers[1:]):
    #     viterbi.append(bp[viterbi[-1]])
    # viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    # return viterbi, viterbi_score


# debug
def debug_tensors(session, model, data, eval_op, batch_size, tag_to_id, verbose=False):
    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, FLAGS.batch_size)
    for x, y, l, seg in zip(xArray, yArray, lArray, segArray):
        fetches = [model.loss, model.logits, model.trans, model.lstm_inputs, model.targets, model.seq_len, model.seq_len_plus1, model.viterbi_sequence]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = [FLAGS.keep_prob]
        feed_dict[model.max_seq_len] = [max(l)]
        loss, logits, trans, lstm_inputs, targets, seq_len, seq_len_plus1, model_viterbi_sequence = session.run(fetches, feed_dict)

        print("model decode:", model_viterbi_sequence.shape, type(model_viterbi_sequence))
        for logits_, l_ in zip(logits, l):
            logits_ = logits_[:l_+1]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)
            viterbi_sequence = viterbi_sequence[1:]
            print("crf decode: ", viterbi_sequence)
            print("model decode: ", list(model_viterbi_sequence[0][1:l_+1]))
            # test_viterbi_decode(logits_, trans)
            # print("test decode: ", new_viterbi_sequence[1:])
            break
        break


# 将测试集的NER结果生成
def ner_generate_results(session, model, data, batch_size, result_file_name, id_to_tag):
    result_file_writer = codecs.open(result_file_name, encoding="utf-8", mode='w')
    result_file_writer.write("char\tgold_label\tpre_label\n")

    temp_file_writer = codecs.open("ner_data/temp.txt", encoding="utf-8", mode='w')
    temp_file_writer.write("char\tgold_label\tpre_label\n")
    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, batch_size)
    for x, y, l, seg, sent in zip(xArray, yArray, lArray, segArray, sentArray):
        fetches = [model.loss, model.logits, model.trans, model.viterbi_sequence]
        feed_dict = dict()
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = [1.0] # evaluate的时候keep设置为1
        feed_dict[model.max_seq_len] = [max(l)]
        loss, logits, trans, decode_res = session.run(fetches, feed_dict)

        for logits_, decode_item, y_, l_, char_ in zip(logits, decode_res, y, l, sent):
            logits_ = logits_[:l_+1]
            y_ = y_[:l_]
            char_ = char_[:l_]
            #crf decode
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)
            viterbi_sequence = viterbi_sequence[1:]

            new_viterbi_sequence = decode_item[1:l_+1]
            for pre_tag, gold_tag, test_char in zip(new_viterbi_sequence, y_, char_):
                result_file_writer.write(str(test_char)+"\t"+str(id_to_tag[int(gold_tag)])+"\t"+str(id_to_tag[int(pre_tag)])+"\n")
            result_file_writer.write("\n")

            for pre_tag, gold_tag, test_char in zip(viterbi_sequence, y_, char_):
                temp_file_writer.write(str(test_char)+"\t"+str(id_to_tag[int(gold_tag)])+"\t"+str(id_to_tag[int(pre_tag)])+"\n")
            temp_file_writer.write("\n")

    result_file_writer.close()
    temp_file_writer.close()


if __name__ == '__main__':
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file)
    dev_sentences = load_sentences(FLAGS.dev_file)
    test_sentences = load_sentences(FLAGS.test_file)

    # create dictionary for word
    if FLAGS.pre_emb:
        # train中的字和词频字典
        dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
        # train+test+pretrain char 字典:dico_chars
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(
            dico_chars_train.copy(),
            FLAGS.vector_file,
            list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences]))
        )

    else:
        _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

    # Create a dictionary and a mapping for tags
    _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    config = get_config()
    config.vocab_size = len(char_to_id)
    eval_config = get_config()
    eval_config.vocab_size = len(char_to_id)

    char_vectors = load_word2vec(FLAGS.vector_file, id_to_char, config.embedding_size)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.seg_scope_name, reuse=None, initializer=initializer):
            m = Segmenter(config=config, init_embedding=char_vectors)
        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.seg_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.seg_train_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        best_f = 0.0

        # debug
        # m.assign_lr(session, config.learning_rate)
        # debug_tensors(session, m, train_data, tf.no_op(), config.batch_size, tag_to_id)
        #  ner_evaluate(session, m, test_data, tf.no_op(), config.batch_size, tag_to_id)

        # train
        for i in range(0, FLAGS.max_epoch):
            m.assign_lr(session, config.learning_rate)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_losses = run_epoch(session, m, train_data, m.train_op, config.batch_size, verbose=True)

            dev_accuracy, dev_total_P, dev_total_R, dev_total_F, dev_per_P, dev_per_R, dev_per_F, dev_loc_P, dev_loc_R, dev_loc_F, \
            dev_org_P, dev_org_R, dev_org_F = ner_evaluate(session, m, dev_data, tf.no_op(), config.batch_size, tag_to_id)
            print("Dev Accuracy: %f, total P:%f, R:%f, F:%f" % (dev_accuracy, dev_total_P, dev_total_R, dev_total_F))
            print("Dev PER P:%f, R:%f, F:%f" % (dev_per_P, dev_per_R, dev_per_F))
            print("Dev LOC P:%f, R:%f, F:%f" % (dev_loc_P, dev_loc_R, dev_loc_F))
            print("Dev ORG P:%f, R:%f, F:%f" % (dev_org_P, dev_org_R, dev_org_F))

            test_accuracy, test_total_P, test_total_R, test_total_F, test_per_P, test_per_R, test_per_F, test_loc_P, test_loc_R, test_loc_F, \
            test_org_P, test_org_R, test_org_F = ner_evaluate(session, m, test_data, tf.no_op(), config.batch_size, tag_to_id)
            print("Test Accuracy: %f, total P:%f, R:%f, F:%f" % (test_accuracy, test_total_P, test_total_R, test_total_F))
            print("Test PER P:%f, R:%f, F:%f" % (test_per_P, test_per_R, test_per_F))
            print("Test LOC P:%f, R:%f, F:%f" % (test_loc_P, test_loc_R, test_loc_F))
            print("Test ORG P:%f, R:%f, F:%f" % (test_org_P, test_org_R, test_org_F))

            if dev_total_F > best_f:
                best_f = dev_total_F
                checkpoint_path = os.path.join(FLAGS.seg_train_dir, "ner_bilstm.ckpt")
                m.saver.save(session, checkpoint_path)
                print("Model Saved...")

        print("Saved model evaluate on test data...")
        test_accuracy, test_total_P, test_total_R, test_total_F, test_per_P, test_per_R, test_per_F, test_loc_P, test_loc_R, test_loc_F, \
        test_org_P, test_org_R, test_org_F = ner_evaluate(session, m, test_data, tf.no_op(), config.batch_size, tag_to_id)
        print("Test Accuracy: %f, total P:%f, R:%f, F:%f" % (test_accuracy, test_total_P, test_total_R, test_total_F))
        print("Test PER P:%f, R:%f, F:%f" % (test_per_P, test_per_R, test_per_F))
        print("Test LOC P:%f, R:%f, F:%f" % (test_loc_P, test_loc_R, test_loc_F))
        print("Test ORG P:%f, R:%f, F:%f" % (test_org_P, test_org_R, test_org_F))

    # 生成测试集的预测结果并将其存储到文件，这里的batch size设置为1
    with tf.Graph().as_default(), tf.Session() as result_session:
        config.batch_size = 1
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.seg_scope_name, reuse=None, initializer=initializer):
            result_m = Segmenter(config=config, init_embedding=char_vectors)
        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.seg_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            result_m.saver.restore(result_session, tf.train.latest_checkpoint(FLAGS.seg_train_dir))
            print("save batch size 1 model")
            checkpoint_path = os.path.join(FLAGS.seg_train_dir, "ner_bilstm.ckpt")
            result_m.saver.save(result_session, checkpoint_path)
        else:
            print("predict with initial parameters.")
            result_session.run(tf.global_variables_initializer())
        print("predict and save test data predict results")
        ner_generate_results(session=result_session, model=result_m, data=test_data, batch_size=1, result_file_name=FLAGS.test_result_file, id_to_tag=id_to_tag)
