# -*- coding: utf-8 -*-
import collections
import codecs

class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        # self.suggestion = list()  # 当bound是true时，匹配的候选集列表
        self.is_word = False  # 字在词语的结尾，is_word=True
        self.tags = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, tags):
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True
        current.tags = tags

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def startAndEnd(self, word):
        '''

        :param word:
        :return: start_boolean, is_word
        '''
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False, False, ""
        return True, current.is_word, current.tags

    def enumerateMatch(self, word, space="_", backward=False):
        results = []
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                s = word[i:j]
                is_start, is_end, tag = self.startAndEnd(s)
                # 判断所截取字符串是否在分词词典和停用词典内
                if is_end:
                    results.append(("".join(s), i, j, tag))
        return results
        # matched = []
        # while len(word) > 1:
        #     if self.search(word):
        #         matched.append(space.join(word[:]))
        #     del word[-1]
        # return matched

    def enumerateMaxMatch(self, words, space="_", backward=False):
        results = []
        n = 0
        while n < len(words):
            matched = False
            for i in range(len(words) - n, 0, -1):
                s = words[n:n + i]
                is_start, is_end, tag = self.startAndEnd(s)
                # 判断所截取字符串是否在分词词典和停用词典内
                if is_end:
                    results.append(("".join(s), n, n + i, tag))
                    matched = True
                    n = n + i
                    break
            if not matched:
                i = 1
                n += i
        return results

    def BDMaxMatch(self, words, space="_", backward=False):
        fw_results = []
        n = 0
        while n < len(words):
            matched = False
            for i in range(len(words) - n, 0, -1):
                s = words[n:n + i]
                is_start, is_end, tag = self.startAndEnd(s)
                # 判断所截取字符串是否在分词词典和停用词典内
                if is_end:
                    fw_results.append(("".join(s), n, n + i, tag))
                    matched = True
                    n = n + i
                    break
            if not matched:
                i = 1
                n += i

        bw_results = []
        n = len(words)
        while n > 0:
            matched = False
            for i in range(n, 0, -1):
                s = words[n - i:n]
                is_start, is_end, tag = self.startAndEnd(s)
                # 判断所截取字符串是否在分词词典和停用词典内
                if is_end:
                    bw_results.append(("".join(s), n - i, n, tag))
                    matched = True
                    n = n - i
                    break
            if not matched:
                i = 1
                n -= i

        if not len(fw_results) == len(bw_results):
            return fw_results if len(fw_results) < len(bw_results) else bw_results

        else:
            fw_set = set(fw_results)
            bw_set = set(bw_results)
            if len(fw_set.symmetric_difference(bw_set)) == 0:
                return fw_results
            else:
                fw_sym = fw_set.difference(bw_set)
                bw_sym = bw_set.difference(fw_set)
                fw_len = sorted([len(element[0]) for element in fw_sym])
                bw_len = sorted([len(element[0]) for element in bw_sym])
                for i in range(len(fw_len)):
                    if fw_len[i] > bw_len[i]:
                        return fw_results
                return bw_results

        return fw_results


class KGTree:
    def __init__(self):
        self.trie = Trie()
        self.ent2type = {}  ## word list to type
        self.space = ""

    def enumerateMatchList(self, word_list):
        match_list = self.trie.enumerateMatch(word_list, self.space)
        return match_list

    def enumerateMaxMatchList(self, word_list):
        match_list = self.trie.enumerateMaxMatch(word_list, self.space)
        return match_list

    def BDMaxMatchList(self, word_list):
        match_list = self.trie.BDMaxMatch(word_list, self.space)
        return match_list

    def search(self, word):
        is_word = self.trie.search(word)
        return is_word

    def insert(self, word_list, source):
        self.trie.insert(word_list, source)
        string = self.space.join(word_list)
        if string not in self.ent2type:
            self.ent2type[string] = source

    def size(self):
        return len(self.ent2type)

    def build(self, kg_file, kgtag_idx):
        kg_tags = {}
        # with open(kg_file) as fp:
        with codecs.open(kg_file, mode='r', encoding="utf-8") as fp:
            for line in fp:
                fields = line.strip().split('\001')
                if len(fields) != 2:
                    continue
                try:
                    entity = fields[0]
                    kgtag_id = str(kgtag_idx[fields[1].strip()])
                except:
                    print(fields)
                    print(line)
                    continue

                tag_ids = kg_tags.get(entity, [])
                if kgtag_id not in tag_ids:
                    tag_ids.append(kgtag_id)
                    kg_tags[entity] = tag_ids

            for entity, tag_ids in list(kg_tags.items()):
                if len(entity) > 1 or '1' in tag_ids:
                    self.insert(entity, ",".join(tag_ids))
            print("Load kg file: ", kg_file, " total size:", self.size())

    def get_processing_kg(self, nkgtags):
        """
            :param vocab_tags: dict[ktag] = idx
            :param kgword_tags: dict[kgword] = [tag list]
            :return: f('[听，冰，雨]') = [[0,0,0,0,0], [1,0,0,0,0],[1,0,0,0,0]] = (list of tag_id_oneHot)
            """
        def f(words):
            kg_onehot_list = [[0.0] * nkgtags for i in range(len(words))]
            to_match_words = []
            for word in words:
                if isinstance(word, unicode):
                    to_match_words.append(word)
                else:
                    to_match_words.append(word.decode("utf-8", 'ignore'))
            # matched_entity = kgtree.enumerateMaxMatchList(to_match_words)
            matched_entity = self.BDMaxMatchList(to_match_words)
            # matched_entity = kgtree.enumerateMaxMatchList([word.decode("utf-8", 'ignore') for word in words])
            for entity in matched_entity:
                kg, start, end, tag_ids = entity
                # print "".join(words),"hit:", kg, start, end, tag_ids
                for i in range(start, end):
                    for tag in tag_ids.split(","):
                        if isinstance(tag, str):
                            tag = int(tag)
                        if i == start:
                            # pdet修改这里。
                            kg_onehot_list[i][tag] = 5.0
                        else:
                            kg_onehot_list[i][tag] = 1.0
            return kg_onehot_list

        return f


    def get_processing_new_kg(self):
        """
            :param vocab_tags: dict[ktag] = idx
            :param kgword_tags: dict[kgword] = [tag list]
            :return: f('[听，冰，雨]') = [[0,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0]] = (list of tag_id_oneHot)
            """
        def f(words):
            new_kg_matches = []
            to_match_words = []
            for word in words:
                if isinstance(word, unicode):
                    to_match_words.append(word)
                else:
                    to_match_words.append(word.decode("utf-8", 'ignore'))
            matched_entity = self.enumerateMatchList(to_match_words)
            for entity in matched_entity:
                kg, start, end, tag_ids = entity
                tags = [int(tag) for tag in tag_ids.split(",")]
                new_kg_matches.append([start, end, tags])

            return new_kg_matches

        return f
