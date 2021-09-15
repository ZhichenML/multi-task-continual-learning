#!/usr/bin/env python
# encoding=utf-8

from inspect import isbuiltin
import json
import random
# from tensorflow.python.keras.backend import dtype
import yaml
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.preprocessing import sequence
import math
import numpy as np
from utils.general_utils import read_file, convert_to_unicode
# from brain import KnowledgeGraph
from utils.constants import *

import copy 

class Vocabulary(object):
    def __init__(self, meta_file=None, allow_unk=0, unk="$UNK$", pad="$PAD$", max_len=None):
        self.voc2id = {}
        self.id2voc = {}
        self.unk = unk
        self.pad = pad
        self.max_len = max_len
        self.allow_unk = allow_unk
        if meta_file:
            with open(meta_file, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = convert_to_unicode(line.strip("\n"))
                    self.voc2id[line] = i
                    self.id2voc[i] = line
            self.size = len(self.voc2id)
            self.oov_num = self.size + 1

    def fit(self, words_list):
        """
        :param words_list: [[w11, w12, ...], [w21, w22, ...], ...]
        :return:
        """
        word_lst = []
        word_lst_append = word_lst.append
        for words in words_list:
            if not isinstance(words, list):
                print(words)
                continue
            for word in words:
                word = convert_to_unicode(word)
                word_lst_append(word)
        word_counts = Counter(word_lst)
        if self.max_num_word < 0:
            self.max_num_word = len(word_counts)
        sorted_voc = [w for w, c in word_counts.most_common(self.max_num_word)]
        self.max_num_word = len(sorted_voc)
        self.oov_index = self.max_num_word + 1
        self.voc2id = dict(zip(sorted_voc, range(1, self.max_num_word + 1)))
        return self

    def _transform2id(self, word):
        word = convert_to_unicode(word)
        if word in self.voc2id:
            return self.voc2id[word]
        elif self.allow_unk:
            return self.voc2id[self.unk]
        else:
            print("==== transform to id error=====")
            print(word)
            raise ValueError("word:{} Not in voc2id, please check".format(word))

    def _transform_seq2id(self, words_list, padding=1):
        # zhichen: add cls to one-hot tag list
        tmp_onehot = np.zeros(self.size, dtype=np.float32)
        tmp_onehot[self._transform2id('O')] = 1.0

        out_ids_list = []
        out_ids_list.append(tmp_onehot)
        out_ids_labels = [] + [self._transform2id('O')] # 'CLS' in the front
        if self.max_len:
            words_list = words_list[:self.max_len]
        for words in words_list:
            out_ids = np.zeros(self.size, dtype=np.float32)
            for w in words:
                out_ids_labels.append(self._transform2id(w))
                out_ids[self._transform2id(w)] = 1.0
            out_ids_list.append(out_ids)
        if padding and self.max_len:
            while len(out_ids_list) < self.max_len+1:
                out_ids_list.append(np.zeros(self.size, dtype=np.float32))
        if padding and self.max_len:
            while len(out_ids_labels) < self.max_len + 1:# max_len + CLS
                out_ids_labels.append(0)
        return out_ids_list, out_ids_labels
    
    def _transform_cates2ont_hot(self, words, padding=0):
        # 将多标签意图转为 one_hot
        out_ids = np.zeros(self.size, dtype=np.float32)
        if len(words) == 0:
            return out_ids
        else:
            out_ids[self._transform2id(words)] = 1.0
        return out_ids, self._transform2id(words)
    

    def _transform_acts2ont_hot(self, words, padding=0):
        # 将多标签意图转为 one_hot
        out_ids = np.zeros(self.size, dtype=np.float32)
        #actions == ['']的情况
        if len(words) == 0 or (len(words)==1 and words[0]==''):
            return out_ids
        for w in words:
            out_ids[self._transform2id(w)] = 1.0
        return out_ids

    def _transform_seq2bert_id(self, words, padding=0):
        out_ids, seq_len = [], 0
        if self.max_len:
            words = words[:self.max_len]
        seq_len = len(words)+1
        # 插入 [CLS]
        out_ids.append(self._transform2id("[CLS]"))
        for w in words:
            out_ids.append(self._transform2id(w))
        mask_ids = [1 for _ in out_ids]
        if padding and self.max_len:
            while len(out_ids) < self.max_len + 1:# max_len + CLS
                out_ids.append(0)
                mask_ids.append(0)
        seg_ids = [0 for _ in out_ids]
        return out_ids, mask_ids, seg_ids, seq_len

    def transform(self, seq_list, is_bert=0):
        if is_bert:
            return [self._transform_seq2bert_id(seq) for seq in seq_list]
        else:
            return [self._transform_seq2id(seq) for seq in seq_list]

    def __len__(self):
        return len(self.voc2id)

class OSDataset(object):
    """catslu数据集，已经处理成domain|intent + actions + slot sequence格式""" 
    def __init__(self, file_path, cfg, has_label=1, is_train=0, with_kg=0, kg_name=None, once=False):
        # 需要传入 ontology字典，将意图，槽位之类的做转换。
        self.file_path = file_path
        self.is_train = is_train
        self.cfg = cfg
        # word 配置
        self.word_vocab = Vocabulary(cfg['bert_dir'] + cfg['bert_vocab'], allow_unk=1, unk='[UNK]', pad='[PAD]', max_len=cfg['max_seq_len'])
        # domain|intent 配置
        self.cates_vocab = Vocabulary(cfg['meta_dir'] + cfg['cates_file'])
        # actions 配置
        self.acts_vocab = Vocabulary(cfg['meta_dir'] + cfg['acts_file'])        
        # tag 配置
        self.tag_vocab = Vocabulary(cfg['meta_dir'] + cfg['tag_file'], max_len=cfg['max_seq_len'])
        # kbert 配置
        self.with_kg = with_kg
        self.kg_name = kg_name

        if has_label:
            self.get_label_dataset(self.with_kg)
        else:
            self.get_unlabel_dataset(self.with_kg)        

    # deprecated
    def get_new_data(self, file_path, keep_old= False):
        # reset the file path for new data and fetch the data
        if keep_old:
            id_set = self.id_set.copy()
            #dataset = self.dataset.copy()

        self.file_path = file_path
        self.get_label_dataset(self.with_kg)

        if keep_old:
            self.id_set.extend(id_set)
            #self.dataset.extend(dataset)

    def token_transform(self, seg, tokens, transform_table):
        """
           处理加入kg的input_query里，知识属性在bert的词表里没有对应的，
           通过transform_table映射成一个在bert词表里存在的单词
           ex. loc在bert的词表里没有，则转换成address
        """
        tokens_id = []
        for i in range(len(tokens)):
            # 不是添加的实体，或者是添加的实体，但是不在table里
            if seg[i] == 0 or seg[i]==1 and tokens[i] not in transform_table:
                tokens_id.append(self.word_vocab._transform2id(tokens[i])) 
            # 是添加的实体，需要映射到table里的词，才可以变成bert词表里有的词
            else: 
                new_representation = transform_table[tokens[i]]
                tokens_id.append(self.word_vocab._transform2id(new_representation)) 
        return tokens_id

    def get_label_dataset(self, with_kg=0):
        # 读取数据集，然后将其转为对应格式
        self.dataset, self.id_set = [], []

        if self.with_kg:
            # Build knowledge graph.
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        for line in read_file(self.file_path, "\1"):
            query, q_arr, tags, cates, acts = line[:5]
            acts = acts.split("\3")
            q_arr = q_arr.split("\3")
            tag_str_list = tags.strip().split("|")
            tags = [[tag for tag in tags.split()] for tags in tag_str_list]
            #self.dataset.append([query, q_arr, cates, acts, tags])
            # q_ids加入了[CLS]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
            cates_ids = self.cates_vocab._transform_cates2ont_hot(cates)
            acts_ids = self.acts_vocab._transform_acts2ont_hot(acts)
            # 没有[CLS]
            tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            if self.with_kg:
                # pos: [0,1,2,2,3,3,4,...] lenght=max_sequence+1
                # vm: [max_sequence+1, max_sequence+1]
                # seg: lenght=max_sequence+1
                # query="梁咏琪的专辑词 ci放点来听听"
                # 当分词器是jieba时
                sent = query
                # # 当分词器是pkuseg时
                # sent = CLS_TOKEN + query
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(sent, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                # example of tokens, lenght=max_sequence+1
                # ['[CLS]', '您', '好', 'show', 'song', '请', '打', '开',
                #  'song', 'album', '音', '乐', 'show', 'program', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

                mask_ids = [1] * src_length + [0]*(len(tokens)-src_length)

                # filter the first CLS
                src_length = src_length - 1
                
                j = 0
                new_labels = []
                # ignore [CLS]
                for i in range(1, len(tokens)):
                    # 是原始query里的字
                    if seg[i] == 0 and tokens[i] != PAD_ID:
                        kg_label = tags_ids[j]
                        j += 1
                     # 是添加的实体
                    elif seg[i] == 1 and tokens[i] != PAD_ID: 
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)
                        # 这里ENT放在第一个位置
                        kg_label[0] = 1.0
                    # 是 PAD
                    else:
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)

                    new_labels.append(kg_label)
                # print(len(new_labels))
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels])

            else:
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids])
    
    def get_unlabel_dataset(self, with_kg=0):
        # Build knowledge graph.
        if self.with_kg:
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        # 读取数据集，这部分数据集只做预测
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\1"):
            query = line[0]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(query, padding=1)
            if self.with_kg:
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(query, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                mask_ids = [1] * src_length + [0]*(self.cfg['max_seq_len']+1-src_length)
                #self.dataset.append([query, tokens])
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm])               
            else:
                #self.dataset.append([query])
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len])

    def get_batch(self, batch_size=None):
        if self.is_train:
            random.shuffle(self.id_set)
        if not batch_size:
            batch_size = self.cfg['batch_size']
        steps = int(math.ceil(float(len(self.id_set)) / batch_size))
        for i in range(steps):
            idx = i * batch_size
            cur_set = self.id_set[idx: idx + batch_size]
            yield zip(*cur_set)

    def __iter__(self):
        for each in self.id_set:
            yield each
    def __len__(self):
        return len(self.id_set)


class OSDataset_CL_once(object):
    """catslu数据集，已经处理成domain|intent + actions + slot sequence格式""" 
    def __init__(self, file_path, cfg, has_label=1, is_train=0, with_kg=0, kg_name=None):
        # 需要传入 ontology字典，将意图，槽位之类的做转换。
        self.file_path = file_path
        self.is_train = is_train
        self.cfg = cfg
        # word 配置
        self.word_vocab = Vocabulary(cfg['bert_dir'] + cfg['bert_vocab'], allow_unk=1, unk='[UNK]', pad='[PAD]', max_len=cfg['max_seq_len'])
        # domain|intent 配置
        self.cates_vocab = Vocabulary(cfg['meta_dir'] + cfg['cates_file'])
        # actions 配置
        self.acts_vocab = Vocabulary(cfg['meta_dir'] + cfg['acts_file'])        
        # tag 配置
        self.tag_vocab = Vocabulary(cfg['meta_dir'] + cfg['tag_file'], max_len=cfg['max_seq_len'])
        # kbert 配置
        self.with_kg = with_kg
        self.kg_name = kg_name

        self.id_set = []
        self.dataset = []

        self.task_setup = np.load(cfg['data_dir'] + 'task_setup.npz')['task_setup']
        # if isinstance(task_setup, list):
        #     self.task_setup = task_setup

        self.get_label_dataset_once()

    

    def get_label_dataset_once(self, with_kg=0):
        # 读取数据集，然后将其转为对应格式
        self.all_dataset, self.all_id_set = {}, {}
  
        if self.with_kg:
            # Build knowledge graph.
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]
        for task in self.task_setup:
        
            self.all_dataset[task], self.all_id_set[task] = [], []
            for line in read_file(self.file_path+task, "\1"):
                query, q_arr, tags, cates, acts = line[:5]
                acts = acts.split("\3")
                q_arr = q_arr.split("\3")
                tag_str_list = tags.strip().split("|")
                tags = [[tag for tag in tags.split()] for tags in tag_str_list]
                #self.all_dataset[task].append([query, q_arr, cates, acts, tags])
                # q_ids加入了[CLS]
                q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
                cates_ids = self.cates_vocab._transform_cates2ont_hot(cates)
                acts_ids = self.acts_vocab._transform_acts2ont_hot(acts)
                # 没有[CLS]
                tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
                if self.with_kg:
                    # pos: [0,1,2,2,3,3,4,...] lenght=max_sequence+1
                    # vm: [max_sequence+1, max_sequence+1]
                    # seg: lenght=max_sequence+1
                    # query="梁咏琪的专辑词 ci放点来听听"
                    # 当分词器是jieba时
                    sent = query
                    # # 当分词器是pkuseg时
                    # sent = CLS_TOKEN + query
                    tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(sent, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                    
                    vm = vm.astype("bool")
                    tokens = self.token_transform(seg, tokens, self.transform_table)
                    # example of tokens, lenght=max_sequence+1
                    # ['[CLS]', '您', '好', 'show', 'song', '请', '打', '开',
                    #  'song', 'album', '音', '乐', 'show', 'program', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

                    mask_ids = [1] * src_length + [0]*(len(tokens)-src_length)

                    # filter the first CLS
                    src_length = src_length - 1
                    
                    j = 0
                    new_labels = []
                    # ignore [CLS]
                    for i in range(1, len(tokens)):
                        # 是原始query里的字
                        if seg[i] == 0 and tokens[i] != PAD_ID:
                            kg_label = tags_ids[j]
                            j += 1
                        # 是添加的实体
                        elif seg[i] == 1 and tokens[i] != PAD_ID: 
                            kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)
                            # 这里ENT放在第一个位置
                            kg_label[0] = 1.0
                        # 是 PAD
                        else:
                            kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)

                        new_labels.append(kg_label)
                    # print(len(new_labels))
                    self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels])

                else:
                    self.all_id_set[task].append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids])
                
                
           

    def next_task(self, task_name, keep_old=False):
        if keep_old:
            self.id_set = self.id_set + self.all_id_set[task_name]
        else:
            self.id_set = self.all_id_set[task_name]

    def get_new_data(self, file_path, keep_old= False):
        # reset the file path for new data and fetch the data
        if keep_old:
            id_set = self.id_set.copy()
            #dataset = self.dataset.copy()

        self.file_path = file_path
        self.get_label_dataset(self.with_kg)

        if keep_old:
            self.id_set.extend(id_set)
            #self.dataset.extend(dataset)

    def token_transform(self, seg, tokens, transform_table):
        """
           处理加入kg的input_query里，知识属性在bert的词表里没有对应的，
           通过transform_table映射成一个在bert词表里存在的单词
           ex. loc在bert的词表里没有，则转换成address
        """
        tokens_id = []
        for i in range(len(tokens)):
            # 不是添加的实体，或者是添加的实体，但是不在table里
            if seg[i] == 0 or seg[i]==1 and tokens[i] not in transform_table:
                tokens_id.append(self.word_vocab._transform2id(tokens[i])) 
            # 是添加的实体，需要映射到table里的词，才可以变成bert词表里有的词
            else: 
                new_representation = transform_table[tokens[i]]
                tokens_id.append(self.word_vocab._transform2id(new_representation)) 
        return tokens_id

    def get_label_dataset(self, with_kg=0):
        # 读取数据集，然后将其转为对应格式
        self.dataset, self.id_set = [], []

        if self.with_kg:
            # Build knowledge graph.
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        for line in read_file(self.file_path, "\1"):
            query, q_arr, tags, cates, acts = line[:5]
            acts = acts.split("\3")
            q_arr = q_arr.split("\3")
            tag_str_list = tags.strip().split("|")
            tags = [[tag for tag in tags.split()] for tags in tag_str_list]
            #self.dataset.append([query, q_arr, cates, acts, tags])
            # q_ids加入了[CLS]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
            cates_ids = self.cates_vocab._transform_cates2ont_hot(cates)
            acts_ids = self.acts_vocab._transform_acts2ont_hot(acts)
            # 没有[CLS]
            tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            if self.with_kg:
                # pos: [0,1,2,2,3,3,4,...] lenght=max_sequence+1
                # vm: [max_sequence+1, max_sequence+1]
                # seg: lenght=max_sequence+1
                # query="梁咏琪的专辑词 ci放点来听听"
                # 当分词器是jieba时
                sent = query
                # # 当分词器是pkuseg时
                # sent = CLS_TOKEN + query
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(sent, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                # example of tokens, lenght=max_sequence+1
                # ['[CLS]', '您', '好', 'show', 'song', '请', '打', '开',
                #  'song', 'album', '音', '乐', 'show', 'program', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

                mask_ids = [1] * src_length + [0]*(len(tokens)-src_length)

                # filter the first CLS
                src_length = src_length - 1
                
                j = 0
                new_labels = []
                # ignore [CLS]
                for i in range(1, len(tokens)):
                    # 是原始query里的字
                    if seg[i] == 0 and tokens[i] != PAD_ID:
                        kg_label = tags_ids[j]
                        j += 1
                     # 是添加的实体
                    elif seg[i] == 1 and tokens[i] != PAD_ID: 
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)
                        # 这里ENT放在第一个位置
                        kg_label[0] = 1.0
                    # 是 PAD
                    else:
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)

                    new_labels.append(kg_label)
                # print(len(new_labels))
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels])

            else:
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids])
    
    def get_unlabel_dataset(self, with_kg=0):
        # Build knowledge graph.
        if self.with_kg:
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        # 读取数据集，这部分数据集只做预测
        self.dataset, self.id_set = [], []
        for line in read_file(self.file_path, "\1"):
            query = line[0]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(query, padding=1)
            if self.with_kg:
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(query, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                mask_ids = [1] * src_length + [0]*(self.cfg['max_seq_len']+1-src_length)
                #self.dataset.append([query, tokens])
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm])               
            else:
                #self.dataset.append([query])
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len])

    def get_batch(self, batch_size=None):
        if self.is_train:
            random.shuffle(self.id_set)
        if not batch_size:
            batch_size = self.cfg['batch_size']
        steps = int(math.ceil(float(len(self.id_set)) / batch_size))
        for i in range(steps):
            idx = i * batch_size
            cur_set = self.id_set[idx: idx + batch_size]
            yield zip(*cur_set)

    def __iter__(self):
        for each in self.id_set:
            yield each
    def __len__(self):
        return len(self.id_set)


class OSDataset_CL_once_sep(object):
    """catslu数据集，已经处理成domain|intent + actions + slot sequence格式""" 
    def __init__(self, file_path, cfg, has_label=1, is_train=0, with_kg=0, kg_name=None, ner_span=False):
        # 需要传入 ontology字典，将意图，槽位之类的做转换。
        self.file_path = file_path
        self.is_train = is_train
        self.cfg = cfg
        # word 配置
        self.word_vocab = Vocabulary(cfg['bert_dir'] + cfg['bert_vocab'], allow_unk=1, unk='[UNK]', pad='[PAD]', max_len=cfg['max_seq_len'])
        # domain|intent 配置
        self.cates_vocab = Vocabulary(cfg['meta_dir'] + cfg['cates_file'])
        #self.cates_vocab_sep {}
        # actions 配置
        self.acts_vocab = Vocabulary(cfg['meta_dir'] + cfg['acts_file'])        
        # tag 配置
        self.tag_vocab = Vocabulary(cfg['meta_dir'] + cfg['tag_file'], max_len=cfg['max_seq_len'])
        # kbert 配置
        self.with_kg = with_kg
        self.kg_name = kg_name
        
        self.ner_span = ner_span
        if ner_span:
            label_list = []
            for tag in self.tag_vocab.voc2id.keys():
                if tag == "O":
                    label_list.append(tag)
                    continue
                label_list.append(tag.split("_")[1])
            
            self.label_list = list(set(label_list))
            print("label_list: ", self.label_list)
            print("len: ", len(self.label_list))
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
            self.id2label = {i: label for i, label in enumerate(self.label_list)}


        self.id_set = []
        
        self.task_setup = np.load(cfg['data_dir'] + 'task_setup.npz')['task_setup']

        self.get_class_map()

        self.get_label_dataset_once()

        self.replay_id_set = []

        

    def get_class_map(self):
        self.task_setup = np.load(self.cfg['data_dir'] + 'task_setup.npz')['task_setup']
         
        all_intents = np.load(self.cfg['data_dir'] + 'all_intents.npz')['all_intents']
        from collections import defaultdict
        self.domains_intents = defaultdict(list)
        for intent in all_intents:
            tmp = intent.split('|')
            self.domains_intents[tmp[0]].append(intent) # enter: {music, opera ...}

        int_cl_class = 0
        self.class_map = {}
        self.inv_class_map = {}
        for dom in self.task_setup:
            for intent in self.domains_intents[dom]:
                self.class_map[intent] = int_cl_class
                self.inv_class_map[int_cl_class] = intent
                int_cl_class += 1
        self.cates_vocab.voc2id = self.class_map
        self.cates_vocab.id2voc = self.inv_class_map
       # for task, intent in domains_intents:
       # self.cates_vocab_sep[]

        
    def get_label_dataset_once(self, with_kg=0):
        def get_one_hot(words, w):
            size = len(words)
            out_ids = np.zeros(size, dtype=np.float32)
            out_ids[np.where(words==w)] = 1.0
            return out_ids
        # 读取数据集，然后将其转为对应格式
        from collections import defaultdict
        self.all_dataset, self.all_id_set = defaultdict(list), defaultdict(list)

        if self.with_kg:
            # Build knowledge graph.
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]
        for task in self.task_setup:
            
            #self.all_id_set[task] = [], []
            line_counter = 0
            for line in read_file(self.file_path+task, "\1"):
                line_counter+=1
                query, q_arr, tags, cates, acts = line[:5]
                acts = acts.split("\3")
                q_arr = q_arr.split("\3")
                tag_str_list = tags.strip().split("|")
                tags = [[tag for tag in tags.split()] for tags in tag_str_list]
                
                # q_ids加入了[CLS]
                q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
                _, cates_ids = self.cates_vocab._transform_cates2ont_hot(cates)
                
                
                #  cates_ids = get_one_hot(self.domains_intents[task], cates)

                acts_ids = self.acts_vocab._transform_acts2ont_hot(acts)
                # 没有[CLS]
                # zhichen: revised the output from one-hot to id, add [CLS] in tags txt
              
                _, tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            
                if self.with_kg:
                    # pos: [0,1,2,2,3,3,4,...] lenght=max_sequence+1
                    # vm: [max_sequence+1, max_sequence+1]
                    # seg: lenght=max_sequence+1
                    # query="梁咏琪的专辑词 ci放点来听听"
                    # 当分词器是jieba时
                    sent = query
                    # # 当分词器是pkuseg时
                    # sent = CLS_TOKEN + query
                    tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(sent, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                    
                    vm = vm.astype("bool")
                    tokens = self.token_transform(seg, tokens, self.transform_table)
                    # example of tokens, lenght=max_sequence+1
                    # ['[CLS]', '您', '好', 'show', 'song', '请', '打', '开',
                    #  'song', 'album', '音', '乐', 'show', 'program', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                    # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

                    mask_ids = [1] * src_length + [0]*(len(tokens)-src_length)

                    # filter the first CLS
                    src_length = src_length - 1
                    
                    j = 0
                    new_labels = []
                    # ignore [CLS]
                    for i in range(1, len(tokens)):
                        # 是原始query里的字
                        if seg[i] == 0 and tokens[i] != PAD_ID:
                            kg_label = tags_ids[j]
                            j += 1
                        # 是添加的实体
                        elif seg[i] == 1 and tokens[i] != PAD_ID: 
                            kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)
                            # 这里ENT放在第一个位置
                            kg_label[0] = 1.0
                        # 是 PAD
                        else:
                            kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)

                        new_labels.append(kg_label)
                    # print(len(new_labels))
                    self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels])

                else:
                    
                    if self.ner_span:
                        from utils.general_utils import get_chunks
                        # set set start_ids, end_ids, subjects
                        tmp = []
                        for ind, tag in enumerate(tags_ids[1:]):
                            
                            onehot = np.zeros(len(self.tag_vocab.id2voc))
                            onehot[tag] = 1
                            tmp.append(onehot)
                        # print(np.shape(tmp))
                        # 加入cls之后，返回的chunks start和end都加 1， 为了和ner_span统一，L763 对id 减 1
                        tmp_subject = list(get_chunks(tmp, self.tag_vocab.voc2id))
                        
                        subjects = []
                        # change the form of chunks
                        for ids in tmp_subject:
                            ids = list(ids)
                            ids[1] = int(ids[1])
                            ids[2] = int(ids[2])
                            subjects.append(ids)
                        
                        start_ids = [0] * len(q_ids)
                        end_ids = [0] * len(q_ids)
                        subjects_id = []
                        for subject in subjects:
                            label = subject[0]
                            start = subject[1]
                            end = subject[2]
                            start_ids[start+1] = self.label2id[label]
                            
                            end_ids[end+1] = self.label2id[label]
                            
                            subjects_id.append((self.label2id[label], start, end))

                        # print(subject)
                        # print(subjects_id)
                        # input("pause")

                        assert len(q_ids) == self.cfg['max_seq_len']+1
                        assert len(mask_ids) == self.cfg['max_seq_len']+1
                        assert len(seg_ids) == self.cfg['max_seq_len']+1
                        assert len(tags_ids) == self.cfg['max_seq_len']+1
                        self.all_id_set[task].append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids, start_ids, end_ids, subjects_id])

                    else:
                        # if line_counter == 77503:
                        #     print(q_arr, tags)

                        assert len(q_ids) == self.cfg['max_seq_len']+1
                        assert len(mask_ids) == self.cfg['max_seq_len']+1
                        assert len(seg_ids) == self.cfg['max_seq_len']+1
                        assert len(tags_ids) == self.cfg['max_seq_len']+1, (np.shape(tags_ids), line_counter)
                        self.all_id_set[task].append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids])
                
                # if line_counter >= 1000:
                #     break

    def next_task(self, task_name, keep_old=False):
        
        if keep_old:
            self.id_set = self.id_set + self.all_id_set[task_name]
        else:
            self.id_set = self.all_id_set[task_name]
        self.task_name = task_name
        #initial_class = [self.class_map[intent] for intent in self.domains_intents[self.task_setup[0]] ]
        #self.n_classes = len(self.initial_class)


    def token_transform(self, seg, tokens, transform_table):
        """
           处理加入kg的input_query里，知识属性在bert的词表里没有对应的，
           通过transform_table映射成一个在bert词表里存在的单词
           ex. loc在bert的词表里没有，则转换成address
        """
        tokens_id = []
        for i in range(len(tokens)):
            # 不是添加的实体，或者是添加的实体，但是不在table里
            if seg[i] == 0 or seg[i]==1 and tokens[i] not in transform_table:
                tokens_id.append(self.word_vocab._transform2id(tokens[i])) 
            # 是添加的实体，需要映射到table里的词，才可以变成bert词表里有的词
            else: 
                new_representation = transform_table[tokens[i]]
                tokens_id.append(self.word_vocab._transform2id(new_representation)) 
        return tokens_id

    def get_label_dataset(self, with_kg=0):
        # 读取数据集，然后将其转为对应格式
        self.id_set = []

        if self.with_kg:
            # Build knowledge graph.
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        for line in read_file(self.file_path, "\1"):
            query, q_arr, tags, cates, acts = line[:5]
            acts = acts.split("\3")
            q_arr = q_arr.split("\3")
            tag_str_list = tags.strip().split("|")
            tags = [[tag for tag in tags.split()] for tags in tag_str_list]
            #self.dataset.append([query, q_arr, cates, acts, tags])
            # q_ids加入了[CLS]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(q_arr, padding=1)
            cates_ids = self.cates_vocab._transform_cates2ont_hot(cates)
            acts_ids = self.acts_vocab._transform_acts2ont_hot(acts)
            # 没有[CLS]
            tags_ids = self.tag_vocab._transform_seq2id(tags, padding=1)
            if self.with_kg:
                # pos: [0,1,2,2,3,3,4,...] lenght=max_sequence+1
                # vm: [max_sequence+1, max_sequence+1]
                # seg: lenght=max_sequence+1
                # query="梁咏琪的专辑词 ci放点来听听"
                # 当分词器是jieba时
                sent = query
                # # 当分词器是pkuseg时
                # sent = CLS_TOKEN + query
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(sent, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                # example of tokens, lenght=max_sequence+1
                # ['[CLS]', '您', '好', 'show', 'song', '请', '打', '开',
                #  'song', 'album', '音', '乐', 'show', 'program', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
                # '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

                mask_ids = [1] * src_length + [0]*(len(tokens)-src_length)

                # filter the first CLS
                src_length = src_length - 1
                
                j = 0
                new_labels = []
                # ignore [CLS]
                for i in range(1, len(tokens)):
                    # 是原始query里的字
                    if seg[i] == 0 and tokens[i] != PAD_ID:
                        kg_label = tags_ids[j]
                        j += 1
                     # 是添加的实体
                    elif seg[i] == 1 and tokens[i] != PAD_ID: 
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)
                        # 这里ENT放在第一个位置
                        kg_label[0] = 1.0
                    # 是 PAD
                    else:
                        kg_label = np.zeros(self.tag_vocab.size, dtype=np.float32)

                    new_labels.append(kg_label)
                # print(len(new_labels))
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels])

            else:
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids])
    
    def get_unlabel_dataset(self, with_kg=0):
        # Build knowledge graph.
        if self.with_kg:
            if self.kg_name == None:
                spo_files = []
            else:
                spo_files = [self.kg_name]
            kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

            # transform_table 配置
            self.transform_table = self.cfg["transform_table"]

        # 读取数据集，这部分数据集只做预测
        self.id_set = []
        for line in read_file(self.file_path, "\1"):
            query = line[0]
            q_ids, mask_ids, seg_ids, seq_len = self.word_vocab._transform_seq2bert_id(query, padding=1)
            if self.with_kg:
                tokens, pos, vm, seg, src_length = kg.add_knowledge_with_vm(query, add_pad=True, max_length=self.cfg['max_seq_len']+1)
                vm = vm.astype("bool")
                tokens = self.token_transform(seg, tokens, self.transform_table)
                mask_ids = [1] * src_length + [0]*(self.cfg['max_seq_len']+1-src_length)
                #self.dataset.append([query, tokens])
                self.id_set.append([q_ids, mask_ids, seg, src_length, tokens, pos, vm])               
            else:
                #self.dataset.append([query])
                self.id_set.append([q_ids, mask_ids, seg_ids, seq_len])

    def get_batch(self, batch_size=None):
        if self.is_train:
            random.shuffle(self.id_set)
        if not batch_size:
            batch_size = self.cfg['batch_size']
        steps = int(math.ceil(float(len(self.id_set)) / batch_size))
        for i in range(steps):
            idx = i * batch_size
            cur_set = self.id_set[idx: idx + batch_size]
            yield zip(*cur_set)

    def __iter__(self):
        for each in self.id_set:
            yield each
    def __len__(self):
        return len(self.id_set)


if __name__ == "__main__":
    cfg_path = "./config/config_kbert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)

    # # 测试无kg的有label数据
    # test_set = OSDataset(cfg['data_dir'] + cfg['test_file'], cfg)  
    # for each in test_set.get_batch(4):
    #     batch = list(each)
    #     q_ids, mask_ids, seg_ids, seq_len, cates_ids, acts_ids, tags_ids = batch  

    # # 测试无kg，无label数据
    # test_set = OSDataset(cfg['data_dir'] + cfg['test_file'], cfg, has_label=0)  
    # for each in test_set.get_batch(4):
    #     batch = list(each)
    #     q_ids, mask_ids, seg_ids, seq_len = batch  

    # # 测试带kg的有label数据
    # test_set = OSDataset(cfg['data_dir'] + cfg['test_file'], cfg, with_kg=1, kg_name=cfg['kg_file'])
    # for each in test_set.get_batch(4):
    #     batch = list(each)
    #     q_ids, mask, seg, src_length, tokens, pos, vm, cates_ids, acts_ids, new_labels = batch

    # 测试带kg的无label数据
    test_set = OSDataset(cfg['data_dir'] + cfg['test_file'], cfg, has_label=0, with_kg=1, kg_name=cfg['kg_file'])
    for each in test_set.get_batch(4):
        batch = list(each)
        q_ids, mask_ids, seg, src_length, tokens, pos, vm = batch
