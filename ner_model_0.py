#!/usr/bin/python
# -*- coding:utf-8 -*-
""" 
SEG for building a LSTM based SEG model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding
from loader import load_sentences
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from data_utils import data_iterator, load_word2vec

import time
import numpy as np
import tensorflow as tf
import sys
import os
import codecs
import pickle
import itertools


pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nlp_proj/
sys.path.append(pkg_path)

file_path = os.path.dirname(os.path.abspath(__file__))  # ../nlp_proj/seg/
data_path = os.path.join(file_path, "ner_data")  # path to find corpus vocab file
train_dir = os.path.join(file_path, "NER_model_0_checkpoint_file")  # path to find model saved checkpoint file

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
flags.DEFINE_string("test_result_file", os.path.join(data_path, "model0_test_predict_result.txt"), "Path for result")
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
        self._input_data = tf.placeholder(tf.int32, [batch_size, None])
        self._targets = tf.placeholder(tf.int32, [batch_size, None])
        self._seq_len = tf.placeholder(tf.int32, [batch_size])
        self._seg_data = tf.placeholder(tf.int32, [batch_size, None])
        self._max_seq_len = tf.placeholder(tf.int32)

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

        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="Dropout")
        inputs = tf.nn.dropout(inputs, self._dropout_keep_prob)

        # 字窗口特征和分词特征
        inputs = tf.reshape(inputs, [batch_size, -1, self.embedding_size + self.seg_embedding_size])
        self._lstm_inputs = inputs
        seg_data_reshape = tf.reshape(self._seg_data, [batch_size, -1, 1])
        seg_data_reshape = tf.cast(seg_data_reshape, dtype=data_type())
        self._loss, self._logits, self._trans, self._seq_len_plus1 = _bilstm_model(inputs, self._targets, self._seq_len, config, seg_data_reshape, self._max_seq_len)

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
        pad_logits = tf.cast(small * tf.ones([batch_size, max_seq_len, 1]), tf.float32)
        new_logits = tf.concat([logits, pad_logits], axis=-1)
        new_logits = tf.concat([start_logits, new_logits], axis=1)
        new_targets = tf.concat([tf.cast(target_num * tf.ones([batch_size, 1]), tf.int32), targets], axis=-1)
        # CRF encode
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(new_logits, new_targets, seq_len+1)
        loss = tf.reduce_mean(-log_likelihood)
    return loss, new_logits, transition_params, seq_len+1


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
        feed_dict[model.dropout_keep_prob] = FLAGS.keep_prob
        feed_dict[model.max_seq_len] = max(l)
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
        feed_dict[model.dropout_keep_prob] = 1.0#evaluate的时候keep设置为1
        feed_dict[model.max_seq_len] = max(l)
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


# debug
def debug_tensors(session, model, data, eval_op, batch_size, tag_to_id, verbose=False):
    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, FLAGS.batch_size)
    for x, y, l, seg in zip(xArray, yArray, lArray, segArray):
        fetches = [model.loss, model.logits, model.trans, model.lstm_inputs, model.targets, model.seq_len, model.seq_len_plus1]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = FLAGS.keep_prob
        feed_dict[model.max_seq_len] = max(l)
        loss, logits, trans, lstm_inputs, targets, seq_len, seq_len_plus1 = session.run(fetches, feed_dict)
        print(np.asarray(x).shape, " ", np.asarray(y).shape, " ", np.asarray(l).shape, " ", np.asarray(seg).shape)
        print("logits shape:", logits.shape)
        print("targets shape:", targets.shape)
        print("seq len:", seq_len)
        print("seq len plus 1:", seq_len_plus1)
        # print("lstm inputs shape:", lstm_inputs.shape)
        print("real len:", l[0])
        # print(lstm_inputs[0][0])
        break


# 将测试集的NER结果生成
def ner_generate_results(session, model, data, batch_size, result_file_name, id_to_tag):
    result_file_writer = codecs.open(result_file_name, encoding="utf-8", mode='w')
    result_file_writer.write("char\tgold_label\tpre_label\n")
    xArray, yArray, lArray, segArray, sentArray = data_iterator(data, batch_size)
    for x, y, l, seg, sent in zip(xArray, yArray, lArray, segArray, sentArray):
        fetches = [model.loss, model.logits, model.trans]
        feed_dict = dict()
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.seq_len] = l
        feed_dict[model.seg_data] = seg
        feed_dict[model.dropout_keep_prob] = 1.0  # evaluate的时候keep设置为1
        feed_dict[model.max_seq_len] = max(l)
        loss, logits, trans = session.run(fetches, feed_dict)

        for logits_, y_, l_, char_ in zip(logits, y, l, sent):
            logits_ = logits_[:l_+1]
            y_ = y_[:l_]
            char_ = char_[:l_]
            #crf decode
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)
            viterbi_sequence = viterbi_sequence[1:]
            for pre_tag, gold_tag, test_char in zip(viterbi_sequence, y_, char_):
                result_file_writer.write(str(test_char)+"\t"+str(id_to_tag[int(gold_tag)])+"\t"+str(id_to_tag[int(pre_tag)])+"\n")
            result_file_writer.write("\n")
    result_file_writer.close()


if __name__ == '__main__':
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

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
        for i in range(FLAGS.max_epoch):
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
        else:
            print("predict with initial parameters.")
            result_session.run(tf.global_variables_initializer())
        print("predict and save test data predict results")
        ner_generate_results(session=result_session, model=result_m, data=test_data, batch_size=1, result_file_name=FLAGS.test_result_file, id_to_tag=id_to_tag)
