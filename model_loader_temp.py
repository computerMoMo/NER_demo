# -*- coding:utf-8 -*-
from loader import load_sentences
from loader import prepare_dataset
from data_utils_for_model1 import data_iterator
import tensorflow as tf
import os
import codecs


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "ner_data")
    check_point_dir_path = "ner_model_cc_debug/"
    # names
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
    test_sentences = load_sentences(os.path.join(data_path, "test_data.txt"))

    # load dictionary
    char_id_file = codecs.open("ner_data/char_to_id.txt", mode='r', encoding='utf-8')
    tag_id_file = codecs.open("ner_data/tag_to_id.txt", mode='r', encoding='utf-8')
    id_tag_file = codecs.open("ner_data/id_to_tag.txt", mode='r', encoding='utf-8')
    char_to_id = dict()
    for line in char_id_file:
        line_list = line.strip().split('\t')
        char_to_id[line_list[0]] = int(line_list[1])

    tag_to_id = dict()
    for line in tag_id_file:
        line_list = line.strip().split('\t')
        tag_to_id[line_list[0]] = int(line_list[1])

    id_to_tag = dict()
    for line in id_tag_file:
        line_list = line.strip().split('\t')
        id_to_tag[int(line_list[0])] = line_list[1]
    # prepare test data
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, lower=True)
    xArray, yArray, lArray, segArray, sentArray = data_iterator(data=test_data, batch_size=1)#batch size=1,和模型保存时保持一致

    # load model and predict
    result_file_writer = codecs.open("ner_data/test_cc_res_temp.txt", encoding="utf-8", mode='w')
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
            loss, logits, trans, crf_decode_res = sess.run(fetches, feed_dict)
            for y_, l_, char_, decode_item in zip(y, l, sent, crf_decode_res):
                y_ = y_[:l_]
                char_ = char_[:l_]
                # crf decode
                temp_viterbi_sequence = decode_item[1:l_+1]
                for pre_tag, gold_tag, test_char in zip(temp_viterbi_sequence, y_, char_):
                    result_file_writer.write(str(test_char) + "\t" + str(id_to_tag[int(gold_tag)]) + "\t" + str(id_to_tag[int(pre_tag)]) + "\n")
                result_file_writer.write("\n")
    result_file_writer.close()

