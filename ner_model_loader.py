# -*- coding:utf-8 -*-
from loader import load_sentences
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from data_utils_for_model1 import data_iterator, load_word2vec

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

if __name__ == '__main__':
    check_point_dir_path = "ner_model_cc_debug/"

    input_data_name = "seg_var_scope/input_data:0"
    targets_name = "seg_var_scope/targets:0"
    seq_len_name = "seg_var_scope/seq_len:0"
    seg_data_name = "seg_var_scope/seg_data:0"
    max_seq_len_name = "seg_var_scope/max_seq_len:0"
    dropout_keep_name = "seg_var_scope/Dropout:0"

    crf_losses_name = "seg_var_scope/loss/crf_losses:0"
    crf_logits_name = "seg_var_scope/loss/crf_logits:0"
    crf_trans_name = "seg_var_scope/loss/transitions:0"
    crf_decode_name = "seg_var_scope/ReverseSequence_1:0"

    # load data sets
    train_sentences = load_sentences(FLAGS.train_file)
    test_sentences = load_sentences(FLAGS.test_file)

    # create dictionary for word
    if FLAGS.pre_emb:
        # train中的字和词频字典
        dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
        # train+test+pre train char 字典:dico_chars
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(
            dico_chars_train.copy(),
            FLAGS.vector_file,
            list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences]))
        )

    else:
        _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

    # Create a dictionary and a mapping for tags
    _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)

    config = get_config()
    config.vocab_size = len(char_to_id)
    eval_config = get_config()
    eval_config.vocab_size = len(char_to_id)

    char_vectors = load_word2vec(FLAGS.vector_file, id_to_char, config.embedding_size)

    xArray, yArray, lArray, segArray, sentArray = data_iterator(data=test_data, batch_size=20)#batch size=20,和模型保存时保持一致

    # data debug
    # for x, y, l, seg, sent in zip(xArray, yArray, lArray, segArray, sentArray):
    #     print("---x---")
    #     print("x shape:", np.asarray(x).shape)
    #     print(x)
    #     print("---y---")
    #     print("y shape:", np.asarray(y).shape)
    #     print(y)
    #     print("---l---")
    #     print("l shape:", np.asarray(l).shape)
    #     print(l)
    #     print("max l:", max(l))
    #     print("---seg---")
    #     print("seg shape:", np.asarray(seg).shape)
    #     print(seg)
    #
    #     break

    result_file_writer = codecs.open("ner_data/test_cc_res.txt", encoding="utf-8", mode='w')
    result_file_writer.write("char\tgold_label\tpre_label\n")
    with tf.Session() as sess:
        # load model
        saver = tf.train.import_meta_graph(check_point_dir_path+'ner_bilstm.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir_path))
        graph = tf.get_default_graph()
        # feed ops
        input_data = graph.get_tensor_by_name(input_data_name)
        targets = graph.get_tensor_by_name(targets_name)
        seq_len = graph.get_tensor_by_name(seq_len_name)
        seg_data = graph.get_tensor_by_name(seg_data_name)
        max_seq_len = graph.get_tensor_by_name(max_seq_len_name)
        dropout_keep = graph.get_tensor_by_name(dropout_keep_name)
        # fetch ops
        crf_losess = graph.get_tensor_by_name(crf_losses_name)
        crf_logits = graph.get_tensor_by_name(crf_logits_name)
        crf_trans = graph.get_tensor_by_name(crf_trans_name)
        crf_decode = graph.get_tensor_by_name(crf_decode_name)
        # session run
        for x, y, l, seg, sent in zip(xArray, yArray, lArray, segArray, sentArray):
            fetches = [crf_losess, crf_logits, crf_trans, crf_decode]
            feed_dict = dict()
            feed_dict[input_data] = x
            feed_dict[targets] = y
            feed_dict[seq_len] = l
            feed_dict[seg_data] = seg
            feed_dict[max_seq_len] = [max(l)]
            feed_dict[dropout_keep] = [1.0]
            # print("x data shape:", np.asarray(x).shape)
            # print("y data shape:", np.asarray(y).shape)
            # print("seq len shape:", np.asarray(l).shape)
            # print("seg data shape:", np.asarray(seg).shape)
            # print("max seq len shape:", np.asarray(max(l)).shape)
            # print(max_seq_len.shape)
            # print("max len:", max(l))
            # print("dropout shape:", )
            loss, logits, trans, crf_decode_res = sess.run(fetches, feed_dict)
            # print(crf_decode_res.shape)
            # print(crf_decode_res)
            # break
            for y_, l_, char_, decode_item in zip(y, l, sent, crf_decode_res):
                y_ = y_[:l_]
                char_ = char_[:l_]
                # crf decode
                temp_viterbi_sequence = decode_item[1:l_+1]
                for pre_tag, gold_tag, test_char in zip(temp_viterbi_sequence, y_, char_):
                    result_file_writer.write(str(test_char) + "\t" + str(id_to_tag[int(gold_tag)]) + "\t" + str(id_to_tag[int(pre_tag)]) + "\n")
                result_file_writer.write("\n")
    result_file_writer.close()

