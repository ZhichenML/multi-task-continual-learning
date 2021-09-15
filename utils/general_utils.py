# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import six
import sys
import logging

# from tensorflow.python.estimator.canned import metric_keys

def open_file(filename, mode='r'):
    if filename.startswith("oss://"):
        import tensorflow as tf
        return tf.gfile.GFile(filename, mode)
    else:
        return open(filename, mode)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s, %s" % (type(text), text))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def read_file(file_:str, splitter:str=None):
    out_arr = []
    with open(file_, encoding="utf-8") as f: 
        out_arr = [x.strip("\n") for x in f.readlines()]
        if splitter:
            out_arr = [x.split(splitter) for x in out_arr]
    return out_arr


def get_chunks(seq, tags):
    """
    nested
    :param seq:  seq.shape = [max_len, ntag] 二维向量，每个字一个0-1向量，有标签的位置上是1，其他为0
    :param tags: tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
    :return:     result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    id2tag = {idx: tag for tag, idx in tags.items()}
    tag_lsit = []
    seq_idx = 0
    # label_seq=self.refine(label_seq)
    while seq_idx < len(seq):
        # is a pad value
        if sum(seq[seq_idx]) == 0:
            seq_idx += 1
            continue
        # print(seq[seq_idx])
        # print(id2tag)
        cate = detecte_start(seq[seq_idx], id2tag)
        # print("cate:")
        # print(cate)
        cate_single = detecte_single(seq[seq_idx], id2tag)
        # print("cate_single:")
        # print(cate_single)
        for each in cate:
            index, tag_name = each # query的B_字的下标，tag_name是标签的名字，不包含B_
            find_head = 0
            for i in range(seq_idx + 1, len(seq)): # i表示下一个字的下标
                if find_tag(seq[i], 'I_'+tag_name, id2tag) == True: # 判断下一个字的标签是否是I_+tag_name
                    find_head += 1 # 表示标签没打完，得继续看下一个字
                elif find_tag(seq[i], 'L_'+tag_name, id2tag) == True: # 判断下一个字的下标是否是L_+tag_name，如果是，表示标签打完了，往tag_lsit中填写就可以了
                    end_idx = i
                    tag_lsit.append((tag_name, str(seq_idx), str(end_idx)))
                    # tag_lsit.append([tag_name, seq_idx, end_idx])
                    break
                else: # 如果是其他标签，都表示标签无法连起来，失败了
                    continue
        for each in cate_single:
            index, tag_name = each # query的B_字的下标，tag_name是标签的名字，不包含B_
            tag_lsit.append((tag_name, str(seq_idx), str(seq_idx)))
            # tag_lsit.append([tag_name, seq_idx, seq_idx])
        seq_idx += 1
    return tag_lsit

def find_tag(label, find_label_name, id2tag):
    '''
    label是某一个字的标签，一个向量，find_label_name是指定的标签，
    该函数判断一个字的标签是否是find_label_name
    如果是返回true
    '''
    for i in range(len(label)):
        if i >= 0 and i < len(id2tag) and id2tag[i] == find_label_name and label[i] == 1:
            return True
    return False
def detecte_start(label, id2tag):
    # label是一个向量，长度是标签的个数，表示一个字能打出哪几种标签，
    # 由于是嵌套NER，一个字可以打出多种标签
    # 该函数判断该字的标签有几个B_开头的
    out = [] # key是index，value是标签除去B_之后的内容
    for i in range(len(label)):
        if i >= 0 and i < len(id2tag) and id2tag[i].startswith('B_'):
            if label[i] == 1:
                out.append([i, id2tag[i].strip().split('B_')[1]])
    return out



def detecte_single(label, id2tag):
    out = []
    for i in range(len(label)):
        if i >= 0 and i < len(id2tag) and id2tag[i].startswith('U_'):
            if label[i] == 1:
                out.append([i, id2tag[i].strip().split('U_')[1]])
    return out



if __name__ == "__main__": 
    get_chunks(seq, tags)

