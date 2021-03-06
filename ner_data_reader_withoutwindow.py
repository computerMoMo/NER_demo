#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reading POS data input_data and target_data

"""Utilities for reading POS train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode

import collections
import sys, os
import codecs
import re
import io
import jieba
jieba.load_userdict("ner_data/seg_dict.txt")#加载自定义字典
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

global UNKNOWN
UNKNOWN = "<OOV>"


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def _ner_read_file(filename):
    sentences = []  # list(list(str))
    tags = []  # list(list(str))
    chars = []
    infile = codecs.open(filename, encoding='utf-8')

    s = []
    t = []
    for line in infile:
        line = line.strip().split(' ')
        if len(line) < 2:
            if len(s) == 0:
                continue
            sentences.append(s)
            tags.append(t)
            s = []
            t = []
        else:
            chars.append(line[0])
            s.append(line[0])
            t.append(line[1])
    return chars, sentences, tags


def _ner_build_vocab(vector_filename, train_path):
    char_to_id = {}
    char_vectors = []
    if not os.path.isfile(vector_filename):
        print("not found vector file")
    else:
        # char unigram dictionary
        infile = codecs.open(vector_filename, encoding='utf-8')
        line = infile.readline()
        vec_len = int(line.split(" ")[-1])
        idx = 0
        for line in infile:
            line = line.strip().split(" ")
            char_to_id[line[0]] = idx
            vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
            char_vectors.append(vector)
            idx += 1

        # 将train data中含有的字但是,vector file 中没有的加入
        train_file = codecs.open(train_path, encoding='utf-8', mode='r')
        for line in train_file:
            line = line.strip().split(' ')
            if len(line) < 2:
                    continue
            else:
                char = line[0]
                if char not in char_to_id:
                    char_to_id[char] = idx
                    idx += 1
                    char_vectors.append(np.zeros([vec_len], dtype=np.float32))

        char_to_id[UNKNOWN] = idx
        char_vectors.append(np.zeros([vec_len], dtype=np.float32))

    char_vectors = np.asarray(char_vectors, dtype=np.float32)

    # tag dictionary
    taglist = ['O', 'B-ORG', 'I-ORG', 'E-ORG', 'S-ORG', 'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC']
    taglist.append(UNKNOWN)
    tag_to_id = dict(zip(taglist, range(len(taglist))))
    return char_to_id, tag_to_id, char_vectors


def _save_vocab(dict, path):
    # save utf-8 code dictionary
    outfile = codecs.open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n"  # unicode
        outfile.write(line)
    outfile.close()


def _read_vocab(path):
    # read utf-8 code
    file = codecs.open(path, encoding='utf-8')
    vocab_dict = {}
    for line in file:
        pair = line.replace("\n", "").split("\t")
        vocab_dict[pair[0]] = int(pair[1])
    return vocab_dict


def load_vocab(data_path):
    char_to_id = _read_vocab(os.path.join(data_path, "char_to_id"))
    tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
    return char_to_id, tag_to_id


def ner_sentence_to_ids(sentence, char_to_id):
    # 产生分词特征
    seg_idx = get_seg_features("".join(sentence))

    sentence.append('<EOS>')
    sentence.append('<EOS>')
    sentence.insert(0, '<BOS>')
    sentence.insert(0, '<BOS>')
    seg_idx.append(0)
    seg_idx.append(0)
    seg_idx.insert(0, 0)
    seg_idx.insert(0, 0)
    char_idx = []
    new_seg_idx = []
    # # 增加长度为5的窗口特征
    # for i in range(2, len(sentence)-2):
    #     for j in range(-2, 3):
    #         if sentence[i+j] in char_to_id:
    #             char_idx.append(char_to_id[sentence[i+j]])
    #         else:
    #             char_idx.append(char_to_id[UNKNOWN])
    #         new_seg_idx.append(seg_idx[i+j])
    for i in range(2, len(sentence)-2):
        if sentence[i] in char_to_id:
            char_idx.append(char_to_id[sentence[i]])
        else:
            char_idx.append(char_to_id[UNKNOWN])
        new_seg_idx.append(seg_idx[i])
    return len(sentence)-4, char_idx, new_seg_idx


def word_ids_to_sentence(data_path, ids):
    tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
    id_to_tag = {id: tag for tag, id in tag_to_id.items()}
    tagArray = [id_to_tag[i] if i in id_to_tag else id_to_tag[0] for i in ids]
    return tagArray


def _ner_file_to_char_ids(filename, char_to_id, tag_to_id):
    _, sentences, tags = _ner_read_file(filename)
    charArray = []
    tagArray = []
    lenArray = []
    segArray = []
    for sentence, tag in zip(sentences, tags):
        l, char_idx, seg_idx = ner_sentence_to_ids(sentence, char_to_id)
        lenArray.append(l)
        charArray.append(char_idx)
        segArray.append(seg_idx)
        tagArray.append([tag_to_id[t] if t in tag_to_id else tag_to_id[UNKNOWN] for t in tag])
    return charArray, tagArray, lenArray, segArray


def ner_load_data(data_path=None, vector_file="ner_vectors.txt"):
    """Load POS raw data from data directory "data_path".
    Args: data_path
    Returns:
      tuple (train_data, valid_data, test_data, vocab_size)
      where each of the data objects can be passed to iterator.
    """

    train_path = os.path.join(data_path, "train_data.txt")
    dev_path = os.path.join(data_path, "dev_data.txt")
    test_path = os.path.join(data_path, "test_data.txt")
    vector_path = os.path.join(data_path, vector_file)

    # NER中暂时不用
    # bigram_path = os.path.join(data_path, "words_for_training")
    # dict_path = os.path.join(data_path, "PinyinDict.txt")

    char_to_id, tag_to_id, char_vectors = _ner_build_vocab(vector_path, train_path)
    # pinyin_dict = _read_pinyin_dict(dict_path)
    # Save char_dict and tag_dict
    _save_vocab(char_to_id, os.path.join(data_path, "char_to_id"))
    _save_vocab(tag_to_id, os.path.join(data_path, "tag_to_id"))
    print("char dictionary size " + str(len(char_to_id)))
    print("tag dictionary size " + str(len(tag_to_id)))

    # train_char, train_tag, train_dict, train_len = _file_to_char_ids(train_path, char_to_id, tag_to_id, pinyin_dict)
    train_char, train_tag, train_len, train_seg = _ner_file_to_char_ids(train_path, char_to_id, tag_to_id)
    print("train dataset: " + str(len(train_char)) + " " + str(len(train_tag)))

    # dev_char, dev_tag, dev_dict, dev_len = _file_to_char_ids(dev_path, char_to_id, tag_to_id, pinyin_dict)
    dev_char, dev_tag, dev_len, dev_seg = _ner_file_to_char_ids(dev_path, char_to_id, tag_to_id)
    print("dev dataset: " + str(len(dev_char)) + " " + str(len(dev_tag)))

    # test_char, test_tag, test_dict, test_len = _file_to_char_ids(test_path, char_to_id, tag_to_id, pinyin_dict)
    test_char, test_tag, test_len, test_seg = _ner_file_to_char_ids(test_path, char_to_id, tag_to_id)
    print("test dataset: " + str(len(test_char)) + " " + str(len(test_tag)))
    vocab_size = len(char_to_id)
    return (train_char, train_tag, train_len, dev_char, dev_tag, dev_len,
            test_char, test_tag, test_len, char_vectors, vocab_size, train_seg, dev_seg, test_seg)


def ner_iterator(char_data, tag_data, len_data, batch_size, seg_data):
    # data中最后一个batch size的数据是缺失的
    data_len = len(char_data)
    batch_len = data_len // batch_size
    lArray = []
    xArray = []
    yArray = []
    segArray = []
    for i in range(batch_len):
        if len(len_data[batch_size * i: batch_size * (i + 1)]) == 0:
            continue
        maxlen = max(len_data[batch_size * i: batch_size * (i + 1)])
        l = np.zeros([batch_size], dtype=np.int32)
        x = np.zeros([batch_size, maxlen], dtype=np.int32)#因为增加了窗口特征,size:5
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        seg = np.zeros([batch_size, maxlen], dtype=np.int32)#分词特征
        l[:len(len_data[batch_size * i:batch_size * (i + 1)])] = len_data[batch_size * i:batch_size * (i + 1)]
        for j, l_j in enumerate(l[:len(len_data[batch_size * i:batch_size * (i + 1)])]):
            x[j][:l_j] = char_data[batch_size * i + j]
            y[j][:l_j] = tag_data[batch_size * i + j]
            seg[j][:l_j] = seg_data[batch_size * i + j]
        lArray.append(l)
        xArray.append(x)
        yArray.append(y)
        segArray.append(seg)
    return xArray, yArray, lArray, segArray


def ner_shuffle(char_data, tag_data, len_data, seg_data):
    char_data = np.asarray(char_data)
    tag_data = np.asarray(tag_data)
    len_data = np.asarray(len_data)
    seg_data = np.asarray(seg_data)
    idx = np.arange(len(len_data))
    np.random.shuffle(idx)

    return char_data[idx], tag_data[idx], len_data[idx], seg_data[idx]


def main():
    """
    Test load_data method and iterator method
    """
    data_path = "ner_data/"
    vector_file = "ner_vectors.txt"
    train_path = os.path.join(data_path, "train_data.txt")
    dev_path = os.path.join(data_path, "dev_data.txt")
    test_path = os.path.join(data_path, "test_data.txt")
    vector_path = os.path.join(data_path, vector_file)

    # NER中暂时不用
    # bigram_path = os.path.join(data_path, "words_for_training")
    # dict_path = os.path.join(data_path, "PinyinDict.txt")

    char_to_id, tag_to_id, char_vectors = _ner_build_vocab(vector_path, train_path)
    # pinyin_dict = _read_pinyin_dict(dict_path)
    # Save char_dict and tag_dict
    _save_vocab(char_to_id, os.path.join(data_path, "char_to_id"))
    _save_vocab(tag_to_id, os.path.join(data_path, "tag_to_id"))
    print("char dictionary size " + str(len(char_to_id)))
    print("tag dictionary size " + str(len(tag_to_id)))

    # train_char, train_tag, train_dict, train_len = _file_to_char_ids(train_path, char_to_id, tag_to_id, pinyin_dict)
    # train_char, train_tag, train_len, train_seg = _ner_file_to_char_ids(train_path, char_to_id, tag_to_id)
    # print("train dataset: " + str(len(train_char)) + " " + str(len(train_tag)))

    # dev_char, dev_tag, dev_dict, dev_len = _file_to_char_ids(dev_path, char_to_id, tag_to_id, pinyin_dict)
    dev_char, dev_tag, dev_len, dev_seg = _ner_file_to_char_ids(dev_path, char_to_id, tag_to_id)
    print("dev dataset: " + str(len(dev_char)) + " " + str(len(dev_tag)))

    # test_char, test_tag, test_dict, test_len = _file_to_char_ids(test_path, char_to_id, tag_to_id, pinyin_dict)
    # test_char, test_tag, test_len, test_seg = _ner_file_to_char_ids(test_path, char_to_id, tag_to_id)
    # print("test dataset: " + str(len(test_char)) + " " + str(len(test_tag)))
    # vocab_size = len(char_to_id)
    # sums = 0
    # for l in dev_len:
    #     sums = sums+l+1
    # print(sums)
    print(sum(dev_len))
    xArray, yArray, lArray, segArray = ner_iterator(dev_char, dev_tag, dev_len, 1, dev_seg)
    char_sums = 0
    for l in lArray:
        char_sums = char_sums+sum(l)
    print(char_sums)

if __name__ == '__main__':
    main()
    # test = [0,1,2,3]
    # print(test[0:5])
