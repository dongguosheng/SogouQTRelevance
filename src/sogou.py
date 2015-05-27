#!/usr/local/op_paas/python2.7/bin/python
# -*- coding: gbk -*-

from util import *
import jieba
import jieba.posseg as pseg
import jieba.analyse
from gensim import corpora, models
import re
from datetime import datetime
from operator import itemgetter

df_dict = {}
df_dict_del_sub = {}
index_dict = {}
index_dict_del_sub = {}
stopwords = set()

def load_stopwords(filename):
    with open(filename) as f:
        for line in f:
            stopwords.add(gbk2unicode(line.rstrip()))

def get_texts(input_file, with_label=False):
    texts = []
    sogou = Sogou()
    query_title_dict = sogou.load_dataset(input_file)
    for query, title_label_list in query_title_dict.items():
        # new doc attempt
        if with_label:
            text = []
            text.extend([w for w in sogou.get_word_list(query) if w not in stopwords])
            for title, label in title_label_list:
                if float(label) <= 7:
                    text.extend([w for w in sogou.get_word_list(title) if w not in stopwords])
                else:
                    pass
                    # texts.append([w for w in sogou.get_word_list(title) if w not in stopwords])
            texts.append(text)
        else:
            texts.append([w for w in sogou.get_word_list(query) if w not in stopwords])
            for title, label in title_label_list:
                texts.append([w for w in sogou.get_word_list(title) if w not in stopwords])
        
    return texts

def gen_word2vec_model(input_file, with_label=True):
    texts = get_texts(input_file, with_label=with_label)
    model = models.Word2Vec(texts, size=50, window=5, min_count=5, workers=4)
    print 'Begin to train Word2Vec ...'
    model.save('../model/word2vec.model')
    print 'Word2Vec train Complete.'

def gen_topic_model(input_file, with_label=False):
    '''
    Train topic models.
    '''
    lsi_topic_num = 200
    lda_topic_num = 100
    texts = get_texts(input_file, with_label=with_label)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # wrapper tfidf to lsi
    print 'Begin to gen tfidf model ...'
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print 'Begin to gen lsi model ...'
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=lsi_topic_num)
    # not use tfidf to lsi
    # lsi_not_tfidf = models.LsiModel(corpus, id2word=dictionary, num_topics=topic_num)
    print 'Begin to gen lda model ...'
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=lda_topic_num)
    dictionary.save('../model/topic_model/dict')
    tfidf.save('../model/topic_model/tfidf_model')
    lsi.save('../model/topic_model/lsi_model_' + str(lsi_topic_num))
    # lsi_not_tfidf.save('./topic_model/lsi_not_tfidf_model')
    lda.save('../model/topic_model/lda_model_' + str(lda_topic_num))
    print 'Topic Model Complete!'

def gen_df(input_file, output_file, del_subtitle=False):
    # query是否算作文档？
    total_doc = 0
    sogou = Sogou()
    query_title_dict = sogou.load_dataset(input_file)
    for query, title_label_list in query_title_dict.items():
        total_doc += 1
        word_set = set()
        word_list = sogou.get_word_list(query)
        for w in word_list:
            word_set.add(w)
        for word in word_set:
            if word not in df_dict:
                df_dict[word] = 1
            else:
                df_dict[word] += 1
        for title, label in title_label_list:
            total_doc += 1
            word_set = set()
            if del_subtitle:
                title, _ = del_sub_title(title)
            word_list = sogou.get_word_list(title)
            for w in word_list:
                word_set.add(w)
            for word in word_set:
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
    with open(output_file, 'w') as f:
        for word, df in df_dict.items():
            f.write(word.encode('gbk') + '\t' + str(df) + '\n')
    print 'TOTAL_DOC: %d' % total_doc

class Sogou(object):
    def __init__(self, train_set='train.txt', test_set='test.txt'):
        self.train_set = train_set
        self.test_set = test_set
        self.doc_total = 38268
        self.dictionary = ''
        self.tfidf_model = ''
        self.lsi_model = ''
        self.lda_model = ''
        self.word2vec_model = ''

    def load_topic_model(self, dict_file, tfidf_model, lsi_model, lda_model):
        self.dictionary = corpora.Dictionary.load(dict_file)
        self.tfidf_model = models.TfidfModel.load(tfidf_model)
        self.lsi_model = models.LsiModel.load(lsi_model)
        self.lda_model = models.LdaModel.load(lda_model) 

    def get_word_list(self, s_unicode, n=2):
        word_list = []
        # 原始的分词
        word_list = [w.word for w in pseg.cut(s_unicode) if w.flag != 'x']
        # n = 2, bigram
        ngram_list = []
        for i in range(2, n+1):
            for j in range(len(word_list)-i+1):
                bigram = ''.join(word_list[j : i+j])
                if bigram not in stopwords:
                    ngram_list.append(bigram)
        
        # end gram
        endgram_list = []
        if len(word_list) > 0:
            endgram_list.append(word_list[-1] + '@')
        if len(word_list) > 1:
            endgram_list.append(word_list[-2] + endgram_list[-1])
        if len(word_list) > 2:
            endgram_list.append(word_list[-3] + endgram_list[-1])
        if len(word_list) > 3:
            endgram_list.append(word_list[-4] + endgram_list[-1])
        
        word_list.extend(endgram_list)
        word_list.extend(ngram_list)
        return word_list

    def load_word2vec_model(self, word2vec_model):
        self.word2vec_model = models.Word2Vec.load(word2vec_model)
    
    def load_df(self, df_file, del_subtitle=False):
        index = 1
        with open(df_file) as f:
            for line in f:
                word, df = line.rstrip().split('\t')
                w_unicode = gbk2unicode(word)
                if del_subtitle:
                    df_dict_del_sub[w_unicode] = int(df)
                    index_dict_del_sub[w_unicode] = index
                else:
                    df_dict[gbk2unicode(word)] = int(df)
                    index_dict[gbk2unicode(word)] = index
                index += 1

    def load_dataset(self, filename):
        query_title_dict = {}
        with open(filename) as f:
            for line in f:
                if len(line.rstrip().split('\t')) != 3:
                    continue
                query, title, label = line.rstrip().split('\t')
                query = gbk2unicode(query)
                title = strQ2B(gbk2unicode(title))
                # query and title to lower case
                query = eng_to_lower(query)
                title = eng_to_lower(title)
                # process str 统一标点符号
                query = process_str(query)
                title = process_str(title)
                # query rewrite
                query = qc(query)
                
                # digit normalization
                
                # print query + '**before**' + title
                query = digit_norm(query, title)
                title = digit_norm(title, query)
                # print query + '**after**' + title

                if query not in query_title_dict:
                    query_title_dict[query] = [(title, label)]
                else:
                    query_title_dict[query].append((title, label))
        return query_title_dict

    def gen_feature(self, input_file, output_file):
        query_title_dict = self.load_dataset(input_file)
        f_out = open(output_file, 'w')
        f_qid = open('../data/test_qid', 'w')
        qid = 0
        for query, title_label_list in query_title_dict.items():
            qid += 1
            for title, label in title_label_list:
                feature_list = self.__get_feature_list(query, title, label)
                # del sub title
                title_new, sub_title = del_sub_title(title)
                feature_list.extend(self.__get_feature_list(query, title_new, label, del_subtitle=True))
                # L2R format
                f_out.write(str(label) + ' qid:' + str(qid) + ' ')
                for i in range(len(feature_list)):
                    f_out.write(str(i+1) + ':' + str(feature_list[i]) + ' ')
                f_out.write('\n')
                # f_out.write('# ' + query.encode('gbk') + '\n')
                if 'test' in input_file:        
                    f_qid.write(str(qid) + '\t' + query.encode('gbk') + '\t' + title.encode('gbk') + '\t' + label + '\n')

        f_out.close()
        f_qid.close()

    def get_tfidf(self, s, del_subtitle=False):
        tfidf_dict = {}
        tf_dict = {}
        # cal tf
        word_list = self.get_word_list(s)
        for w in word_list:
            # stopwords
            if w in stopwords:
                continue
            if w not in tf_dict:
                tf_dict[w] = 1
            else:
                tf_dict[w] += 1
        # cal tfidf
        tf_max = 0
        alpha = 0.2
        for _, tf in tf_dict.items():
            if tf > tf_max:
                tf_max = tf
        for word, tf in tf_dict.items():
            if del_subtitle:
                tfidf_dict[index_dict_del_sub[word]] = (alpha + (1-alpha) * tf / float(tf_max)) * math.log(self.doc_total / float(df_dict_del_sub[word] + 1))
            else:
                tfidf_dict[index_dict[word]] = (alpha + (1-alpha) * tf / float(tf_max)) * math.log(self.doc_total / float(df_dict[word] + 1))

        return (tfidf_dict, tf_dict)

    def get_bm25(self, tf_dict_q, tf_dict_t, del_subtitle=False):
        '''
        Get BM25 score.
        '''
        score = 0.0
        avgdl = 6.0 if del_subtitle else 10.0
        k1 = 1.2 # [1.2, 2.0]
        b = 0.75
        for word in tf_dict_q:
            tf_t = 0.0
            if word not in tf_dict_t:
                continue
            else:
                tf_t = tf_dict_t[word]
            if del_subtitle:
                score += math.log(self.doc_total / float(df_dict_del_sub[word] + 1)) * tf_t * (k1 + 1) / (tf_t + k1 * (1 - b + b * len(tf_dict_t) / avgdl))
            else:
                score += math.log(self.doc_total / float(df_dict[word] + 1)) * tf_t * (k1 + 1) / (tf_t + k1 * (1 - b + b * len(tf_dict_t) / avgdl))

        return score
    
    def get_keywords(self, tfidf_dict, k):
        '''
        Get Top K from TFIDF Dict as Keywords. jieba textrank bug.
        '''
        rs_dict = {}
        tfidf_list = sorted(tfidf_dict.iteritems(), key=itemgetter(1), reverse=True)
        i = 0
        for k, v in tfidf_list:
            if i < k:
                rs_dict[k] = v
                i += 1
            else:
                break

        return rs_dict
    
    def __get_feature_list(self, query, title, label, del_subtitle=False):
        '''
        Generate features for query-title pair.
        '''
        feature_list = []
        # === before segmentation ===
        # --- query len and title len (有用)
        feature_list.extend([len(query), len(title)])   # 1, 2
        # --- abs_len (基本没用)
        feature_list.append(abs(len(query) - len(title)))   # 3
        # --- q in t? (有点用)
        feature_list.append(1 if query in title else 0) # 4
        # --- t in q? (有点用)
        feature_list.append(1 if title in query else 0) # 5
        # --- edit_distance (作用很小)
        ed = edit_dist(query, title)
        feature_list.append(ed) # 6
        # --- 1 - edit_distance / max len(query, title) (有点用)
        feature_list.append(1 - float(ed) / max(len(query), len(title)))    # 7
        # --- lcs seq (作用很小)
        lcs_seq = lcseq(query, title)   
        feature_list.append(lcs_seq)    # 8
        # --- lcs str (竟然没用)
        lcs_str = lcstr(query, title)
        feature_list.append(lcs_str)    # 9
        # ---
        feature_list.append(float(lcs_seq) / (len(query) + len(title) - lcs_seq))   # 10
        feature_list.append(float(lcs_str) / (len(query) + len(title) - lcs_str))   # 11
        # === segment query, title ===
        tfidf_dict_q, tf_dict_q = self.get_tfidf(query, del_subtitle=del_subtitle)
        tfidf_dict_t, tf_dict_t = self.get_tfidf(title, del_subtitle=del_subtitle)

        # --- bm25 score
        feature_list.append(self.get_bm25(tf_dict_q, tf_dict_t, del_subtitle=del_subtitle))
        
        # --- number of words (基本没作用)
        feature_list.extend([len(tfidf_dict_q), len(tfidf_dict_t)]) # 12, 13
        
        # --- abs number of words (作用很小)
        feature_list.append(abs(len(tfidf_dict_q) - len(tfidf_dict_t))) # 14
        
        # --- jaccard sim (作用很小)
        query_set = set([word for word, _ in tfidf_dict_q.items()])
        title_set = set([word for word, _ in tfidf_dict_t.items()])
        # print tfidf_dict_q
        # print tfidf_dict_t
        feature_list.append(jaccard_sim(query_set, title_set))  # 15
        
        # --- query和title中相同词的个数 (作用很小)
        feature_list.append(len(query_set.intersection(title_set))) # 16
        
        # --- tfidf cosine sim
        feature_list.append(cos_sim(tfidf_dict_q, tfidf_dict_t))  # 17
        
        # --- keywords similarity, keywords+TFIDF (基本没作用，不知道是不是没用好)
        topK = 3
        # keywords_q_dict = self.get_keywords(tfidf_dict_q, topK)
        # keywords_t_dict = self.get_keywords(tfidf_dict_t, topK)
        # feature_list.append(cos_sim(keywords_q_dict, keywords_t_dict))  # 18

        # keywords_q_set = set([word for word, _ in keywords_q_dict.items()])
        # keywords_t_set = set([word for word, _ in keywords_t_dict.items()])
        # feature_list.append(jaccard_sim(keywords_q_set, keywords_t_set))    # 19
        
        # --- tfidf->lsi cosine sim
        query_seg_rs = self.get_word_list(query)
        title_seg_rs = self.get_word_list(title)
        # two different way to gen lsi vector, wrap tfidf or not
        lsi_sim = cos_sim_tm(self.lsi_model[self.tfidf_model[self.dictionary.doc2bow(query_seg_rs)]], self.lsi_model[self.tfidf_model[self.dictionary.doc2bow(title_seg_rs)]])
        # -- lsi_sim = cos_sim_tm(self.lsi_not_tfidf_model[self.dictionary.doc2bow(query_seg_rs)], self.lsi_not_tfidf_model[self.dictionary.doc2bow(title_seg_rs)])
        feature_list.append(lsi_sim)    # 20

        # --- lda cosine sim
        lda_sim = cos_sim_tm(self.lda_model[self.dictionary.doc2bow(query_seg_rs)], self.lda_model[self.dictionary.doc2bow(title_seg_rs)])
        feature_list.append(lda_sim)    # 21

        # --- word2vec similarity   # 22    (作用不明显，如何正确使用还待研究)
        keyerror_num = 0
        if len(query_seg_rs) == 0 or len(title_seg_rs) == 0:
            feature_list.append(0)
        else:
            try:
                feature_list.append(self.word2vec_model.n_similarity(query_seg_rs, title_seg_rs))
            except KeyError:
                keyerror_num += 1
                # print query_seg_rs
                # print title_seg_rs
                feature_list.append(0)

        # print feature_list
        return feature_list

def main():
    now = datetime.now()
    data_dir = '../data/'
    model_dir = '../model/'

    gen_digit_dict()
    load_digit_dict('../data/digit_dict')
     
    load_stopwords(data_dir + 'stopwords.txt')

    gen_df(data_dir + 'data.txt', data_dir + 'df_file')
    gen_df(data_dir + 'data.txt', data_dir + 'df_file_del_sub', del_subtitle=True)
    
    gen_topic_model(data_dir + 'data_for_tm.txt', with_label=True)
    gen_word2vec_model(data_dir + 'data_for_tm.txt', with_label=True)

    print 'DF and models Complete. Cost: ' + str(datetime.now() - now)
    now = datetime.now()

    is_off = False
    tail = '_off' if is_off else ''
    train_set = 'train' + tail + '.txt'
    test_set = 'test' + tail + '.txt'

    sogou = Sogou(train_set=data_dir + train_set, test_set=data_dir + test_set)
    sogou.load_df(data_dir + 'df_file')
    sogou.load_df(data_dir + 'df_file_del_sub', del_subtitle=True)
    sogou.load_topic_model(model_dir + './topic_model/dict', model_dir + './topic_model/tfidf_model', model_dir + './topic_model/lsi_model_200', model_dir + './topic_model/lda_model_100')
    sogou.load_word2vec_model(model_dir + 'word2vec.model')
    
    sogou.gen_feature(sogou.train_set, data_dir + 'train_feature')
    sogou.gen_feature(sogou.test_set, data_dir + 'test_feature')
    print 'Gen Feature Complete. Cost: ' + str(datetime.now() - now)

if __name__ == '__main__':
    main()
