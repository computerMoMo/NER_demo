#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
SEG for building a LSTM based SEG model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys, os

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nlp_proj/
sys.path.append(pkg_path)

import ner_data_reader as reader

file_path = os.path.dirname(os.path.abspath(__file__))  # ../nlp_proj/seg/
data_path = os.path.join(file_path, "ner_data")  # path to find corpus vocab file
train_dir = os.path.join(file_path, "ner_ckpt")  # path to find model saved checkpoint file

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("seg_data_path", data_path, "data_path")
flags.DEFINE_string("seg_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("seg_scope_name", "seg_var_scope", "Define SEG Variable Scope Name")

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

        # Define input and target tensors
        self._input_data = tf.placeholder(tf.int32, [batch_size, None])
        self._targets = tf.placeholder(tf.int32, [batch_size, None])
        # self._dicts = tf.placeholder(tf.float32, [batch_size, None])
        self._seq_len = tf.placeholder(tf.int32, [batch_size])

        with tf.device("/cpu:0"):
            if init_embedding is None:
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
            else:
                self.embedding = tf.Variable(init_embedding, name="embedding", dtype=data_type())
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        inputs = tf.nn.dropout(inputs, config.keep_prob)
        inputs = tf.reshape(inputs, [batch_size, -1, self.embedding_size])  # 比原来sentence长度扩大了9倍
        # d = tf.reshape(self._dicts, [batch_size, -1, 16])

        self._loss, self._logits, self._trans = _bilstm_model(inputs, self._targets, self._seq_len, config)

        with tf.variable_scope("train_ops") as scope:
            # Gradients and SGD update operation for training the model.
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
            self.optimizer = tf.train.AdamOptimizer(self._lr)
            self._train_op = self.optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    # @property
    # def dicts(self):
    #     return self._dicts

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


# seg model configuration, set target num, and input vocab_size
class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.05
    learning_rate = 0.005
    max_grad_norm = 5
    hidden_size = 150
    embedding_size = 100
    max_epoch = 5
    max_max_epoch = 10
    stack = False
    keep_prob = 0.8  # There is one dropout layer on input tensor also, don't set lower than 0.9
    lr_decay = 1 / 1.15
    batch_size = 128  # single sample batch
    vocab_size = 16116
    target_num = 13  # SEG tagging tag number for ChineseNER
    bi_direction = True  # LSTM or BiLSTM


def get_config():
    return LargeConfigChinese()


def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0,
                                        state_is_tuple=True,
                                        reuse=tf.get_variable_scope().reuse)


def _bilstm_model(inputs, targets, seq_len, config):
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
        # if stack 又添加一层双向LSTM???
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
        # output = tf.concat(values=[output, dicts], axis=2)  # 加入了pinying 字典特征
        output = tf.reshape(output, [-1, 2 * hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, target_num], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())

        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits, [batch_size, -1, target_num])
    # CRF层
    with tf.variable_scope("loss") as scope:
        # CRF log likelihood
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, targets, seq_len)
        loss = tf.reduce_mean(-log_likelihood)
    return loss, logits, transition_params


def run_epoch(session, model, char_data, tag_data, len_data, eval_op, batch_size, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    losses = 0.0
    iters = 0.0

    char_data, tag_data, len_data = reader.ner_shuffle(char_data, tag_data, len_data)
    xArray, yArray, lArray = reader.ner_iterator(char_data, tag_data, len_data, batch_size)

    for x, y, l in zip(xArray, yArray, lArray):
        fetches = [model.loss, model.logits, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        # feed_dict[model.dicts] = d
        feed_dict[model.seq_len] = l
        loss, logits, _ = session.run(fetches, feed_dict)
        losses += loss
        iters += 1

        if verbose and iters % 50 == 0:
            print("%.3f perplexity: %.3f" %
                  (iters / float(len(xArray)), np.exp(losses / iters / len(xArray))))

    return np.exp(losses / iters)


def ner_evaluate(session, model, char_data, tag_data, len_data, eval_op, batch_size, verbose=False):
    correct_labels = 0
    total_labels = 0

    xArray, yArray, lArray = reader.ner_iterator(char_data, tag_data, len_data, batch_size)

    per_yp_wordnum = 0
    per_yt_wordnum = 0
    per_cor_num = 0
    org_yp_wordnum = 0
    org_yt_wordnum = 0
    org_cor_num = 0
    loc_yp_wordnum = 0
    loc_yt_wordnum = 0
    loc_cor_num = 0

    for x, y, l in zip(xArray, yArray, lArray):
        fetches = [model.loss, model.logits, model.trans]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        loss, logits, trans = session.run(fetches, feed_dict)

        for logits_, y_, l_ in zip(logits, y, l):
            logits_ = logits_[:l_]
            y_ = y_[:l_]

            #crf decode
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)

            per_yp_wordnum += viterbi_sequence.count(7) + viterbi_sequence.count(8)
            per_yt_wordnum += (y_ == 7).sum() + (y_ == 8).sum()
            org_yp_wordnum += viterbi_sequence.count(3) + viterbi_sequence.count(4)
            org_yt_wordnum += (y_ == 3).sum() + (y_ == 4).sum()
            loc_yp_wordnum += viterbi_sequence.count(11) + viterbi_sequence.count(12)
            loc_yt_wordnum += (y_ == 11).sum() + (y_ == 12).sum()
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += l_

            start = 0
            for i in range(0, len(y_)):
                # 计算PER
                if y_[i] == 7 or y_[i] == 8:
                    flag = True
                    for j in range(start, i + 1):
                        if y_[j] != viterbi_sequence[j]:
                            flag = False
                            break
                    if flag:
                        per_cor_num += 1
                    start = i + 1
                # 计算ORG
                elif y_[i] == 3 or y_[i] == 4:
                    flag = True
                    for j in range(start, i + 1):
                        if y_[j] != viterbi_sequence[j]:
                            flag = False
                            break
                    if flag:
                        org_cor_num += 1
                    start = i + 1
                # 计算LOC
                elif y_[i] == 11 or y_[i] == 12:
                    flag = True
                    for j in range(start, i+1):
                        if y_[j] != viterbi_sequence[j]:
                            flag = False
                            break
                    if flag:
                        loc_cor_num += 1
                    start = i + 1
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


if __name__ == '__main__':
    if not FLAGS.seg_data_path:
        raise ValueError("No data files found in 'data_path' folder")

    print("Begin Loading..")

    raw_data = reader.ner_load_data(FLAGS.seg_data_path)
    train_char, train_tag, train_len, dev_char, dev_tag, dev_len, test_char, test_tag, test_len, char_vectors, vocab_size = raw_data

    config = get_config()
    eval_config = get_config()
    # eval_config.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope(FLAGS.seg_scope_name, reuse=None, initializer=initializer):
            m = Segmenter(config=config, init_embedding=char_vectors)
        # with tf.variable_scope(FLAGS.seg_scope_name, reuse=True, initializer=initializer):
        #  mvalid = Segmenter(is_training=False, config=eval_config, init_embedding=char_vectors)
        #  mtest = Segmenter(is_training=False, config=eval_config, init_embedding=char_vectors)

        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.seg_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.seg_train_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        best_f = 0.0

        for i in range(config.max_max_epoch):
            # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            # m.assign_lr(session, config.learning_rate * lr_decay)
            m.assign_lr(session, config.learning_rate)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_char, train_tag, train_len, m.train_op,
                                         config.batch_size, verbose=True)
            dev_accuracy, dev_total_P, dev_total_R, dev_total_F, dev_per_P, dev_per_R, dev_per_F, dev_loc_P, dev_loc_R, dev_loc_F, \
            dev_org_P, dev_org_R, dev_org_F = ner_evaluate(session, m, dev_char, dev_tag, dev_len, tf.no_op(), config.batch_size)
            print("Dev Accuray: %f, total P:%f, R:%f, F:%f" % (dev_accuracy, dev_total_P, dev_total_R, dev_total_F))
            print("Dev PER P:%f, R:%f, F:%f" % (dev_per_P, dev_per_R, dev_per_F))
            print("Dev LOC P:%f, R:%f, F:%f" % (dev_loc_P, dev_loc_R, dev_loc_F))
            print("Dev ORG P:%f, R:%f, F:%f" % (dev_org_P, dev_org_R, dev_org_F))

            if dev_total_F > best_f:
                test_accuracy, test_total_P, test_total_R, test_total_F, test_per_P, test_per_R, test_per_F, test_loc_P, test_loc_R, test_loc_F, \
                test_org_P, test_org_R, test_org_F = ner_evaluate(session, m, test_char, test_tag, test_len, tf.no_op(), config.batch_size)
                print("Test Accuray: %f, total P:%f, R:%f, F:%f" % (test_accuracy, test_total_P, test_total_R, test_total_F))
                print("Test PER P:%f, R:%f, F:%f" % (test_per_P, test_per_R, test_per_F))
                print("Test LOC P:%f, R:%f, F:%f" % (test_loc_P, test_loc_R, test_loc_F))
                print("Test ORG P:%f, R:%f, F:%f" % (test_org_P, test_org_R, test_org_F))

                best_f = dev_total_F
                checkpoint_path = os.path.join(FLAGS.seg_train_dir, "ner_bilstm.ckpt")
                m.saver.save(session, checkpoint_path)
                print("Model Saved...")

        test_accuracy, test_total_P, test_total_R, test_total_F, test_per_P, test_per_R, test_per_F, test_loc_P, test_loc_R, test_loc_F, \
        test_org_P, test_org_R, test_org_F = ner_evaluate(session, m, test_char, test_tag, test_len, tf.no_op(),
                                                          config.batch_size)
        print("Test Accuray: %f, total P:%f, R:%f, F:%f" % (test_accuracy, test_total_P, test_total_R, test_total_F))
        print("Test PER P:%f, R:%f, F:%f" % (test_per_P, test_per_R, test_per_F))
        print("Test LOC P:%f, R:%f, F:%f" % (test_loc_P, test_loc_R, test_loc_F))
        print("Test ORG P:%f, R:%f, F:%f" % (test_org_P, test_org_R, test_org_F))
