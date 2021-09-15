# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import jieba.posseg as pseg
import pkuseg
import numpy as np
from utils.constants import *


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        # 分词器是pkuseg
        # self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        # 分词器是jieba
        self.tokenizer = pseg
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, obje = line.strip().split("\1")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch  e.g., "abcd"
        return: know_sent - sentences with entites embedding
                position - position index of each character.
                visible_matrix -  visible matrixs
                seg - segment tags
        """
        # different tokenizer API could influence the result of entity tag
        # #分词器是pkuseg
        # split_sent = self.tokenizer.cut(sent)

        # 分词器是jieba
        words = self.tokenizer.cut(sent)
        split_sent = []
        for w, t in words:
            split_sent.append(w)
        split_sent = [CLS_TOKEN] + split_sent

        # create tree
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        for token in split_sent:
            entities = list(self.lookup_table.get(token, []))[:max_entities]
            sent_tree.append((token, entities))

            if token in self.special_tags:
                token_pos_idx = [pos_idx+1]
                token_abs_idx = [abs_idx+1]
            else:
                token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_pos_idx = [token_pos_idx[-1] + 1]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + 1]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []
        pos = []
        seg = []
        #sent_tree: [('现在', ['abook_album', 'song']), ('放', []), ('首歌', ['song']), ('来听', ['abook_program'])]

        # pos_idx_tree 对应soft position, 用于构造seg, pos
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in self.special_tags:
                know_sent += [word]
                seg += [0]
            else:
                add_word = list(word)
                know_sent += add_word 
                seg += [0] * len(add_word)
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                add_word = sent_tree[i][1][j]
                know_sent += [add_word]
                seg += [1]
                pos += pos_idx_tree[i][1][j]
        
        token_num = len(know_sent)

        # abs_idx_tree 对应于绝对的位置，用于构造visual matrix
        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [config.PAD_TOKEN] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]
            src_length = max_length
        
        return know_sent, pos, visible_matrix, seg, src_length

