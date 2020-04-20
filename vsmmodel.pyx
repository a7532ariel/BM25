import os
import re
import math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import logging
import time
from tqdm import tqdm
cimport cython

file_list_path = './model/file-list'
vocab_all_path = './model/vocab.all'
inverted_file_path = './model/inverted-file'
news_path = './CIRB010'

cdef int num_docs = 46972
cdef int avg_len = 688
cdef float k = 1.5
cdef float b = 0.75
cdef float k_q = 100
cdef int alpha = 1
cdef float beta = 0.75
cdef float gamma = 0.15

loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
r = [',',
 '.',
 '?',
 '!',
 '=',
 '+',
 '(',
 ')',
 '*',
 ':',
 ';',
 '_',
 '、',
 '＠',
 '～',
 '「',
 '’',
 '＾',
 '＊',
 '“',
 ' ',
 '【',
 '!',
 '』',
 '\\',
 '-',
 '\\',
 '|',
 '（',
 '『',
 '！',
 '<',
 ' ',
 '/',
 '＃',
 '?',
 ' ',
 '？',
 '；',
 '”',
 ' ',
 ',',
 '＄',
 '＝',
 '，',
 '‘',
 '。',
 '】',
 '）',
 '：',
 '》',
 '＆',
 '>',
 '％',
 '＋',
 '\\']

class Retriever:
    
    def __init__(self, args):
        super().__init__()
        
        self.model_dir = args.model_dir
        self.corpus_dir = args.corpus_dir
        self.query_file = args.query_file
        
        self.dl = np.load('dl.npy')
        cdef dict cidf_calculate = {}
        self.idf_calculate = cidf_calculate
        cdef list ctf_calculate = [{} for i in range(num_docs)]
        self.tf_calculate = ctf_calculate
        cdef list ctf = [{} for i in range(num_docs)]
        self.tf = ctf
        
        
        self.read_vocab_all()
        self.read_corpus()
        self.build_dict()



    def read_vocab_all(self):
        logging.info('read_vocab_all...')
        cdef dict cword2id = {}
        cdef int i
        self.word2id = cword2id
        
        with open(os.path.join(self.model_dir, 'vocab.all'), 'r') as f:
            file_list = [line.rstrip('\n') for line in f][1:]
               
        self.word2id['-1'] = -1
        i = 1
        for fn in file_list:
            self.word2id[fn] = i
            i += 1
        cdef list cr_id = [] 
        self.r_id = cr_id
        for s in r:
            if s in self.word2id:
                self.r_id.append(self.word2id[s])
        logging.info('Done!') 

    def read_corpus(self):
        logging.info('read_corpus...')
        cdef dict cfilename2id = {}
        cdef list cid2filename = []
        self.filename2id = cfilename2id
        self.id2filename = cid2filename
        file_path = []
        with open(os.path.join(self.model_dir, 'file-list'), 'r') as f:
            file_list = [line.rstrip('\n') for line in f]

        for fname in tqdm(file_list):
            f_id = fname[-15:].lower()
            self.filename2id[f_id] = len(self.id2filename)
            self.id2filename.append(f_id)

    def cut_text(self, text):
        text = text.replace('\n', '')
        text = text.replace(' ', '')
        text = re.sub(r'[,.?!=+()*:;_、＠～「’＾＊“ 【!』\-\|（『！< /＃? ？；” ,＄＝，‘。】）：》＆>％＋\\」]', '', text)
        return text

    def build_dict(self):
        logging.info('build dict...')
        cdef dict cbigram = {}
        cdef int i, line_num, total_line, article_idx, article_id
        cdef str line
        cdef dict cidf = {}
        cdef list ids = []
        cdef int idf
        self.bigram = cbigram
        with open(os.path.join(self.model_dir, 'inverted-file'), 'r') as f:
            inverted_file = [line.rstrip('\n').split(' ') for line in f]
        total_line = len(inverted_file)
        i = 0

        self.idf = cidf
        line_num = 0
        while line_num < total_line:
            ids = inverted_file[line_num]
            if len(ids) == 3:
                if int(ids[0]) in self.r_id or int(ids[1]) in self.r_id:
                    line_num += (int(ids[2]) + 1)
                    continue
                
                self.bigram['{}_{}'.format(int(ids[0]), int(ids[1]))] = i
            
                  
                idf = int(ids[2])
                self.idf[i] = np.log((num_docs - idf + 0.5)/(idf + 0.5))
                for article_idx in range(idf):
                    line_num += 1
                    ids = inverted_file[line_num]
                    article_id = int(ids[0])
                    self.tf[article_id][i] = int(ids[1])
                line_num += 1
                i += 1
        
        self.denominator_term = np.zeros([num_docs,])
        for i in range(num_docs):
            self.denominator_term[i] = k * (1 - b + b * self.dl[i] / avg_len)

        logging.info('Done!') 


    def retrieve(self):
        cdef int n, i,j, index
        cdef int term, uniterm
        logging.info('Retrieve..!')
        scores = []
        root = ET.parse(self.query_file).getroot()

        origin_queries = []
        for query_node in root:
            tStart = time.time()
            queries = []
            query_terms = {}
            queries.append(query_node.findall('.//concepts')[0].text.replace('\n', ''))
            queries.append(query_node.findall('.//narrative')[0].text.replace('\n', ''))
            for i in range(len(queries)):
                if i == 0:
                    n = 3
                else:
                    n = 1
                queries[i] = ''.join(self.cut_text(queries[i])) # rm stop word and combine 
                t1 = self.word2id[queries[i][0]]
                
                query_terms[self.bigram['{}_{}'.format(t1, -1)]] = 1 * n
                for j in range(len(queries[i])-1):
                    if queries[i][j] in self.word2id and queries[i][j+1] in self.word2id:
                        if '{}_{}'.format(self.word2id[queries[i][j]], self.word2id[queries[i][j+1]]) in self.bigram:
                            term = self.bigram['{}_{}'.format(self.word2id[queries[i][j]], self.word2id[queries[i][j+1]])]
                            if term in query_terms:
                                query_terms[term] += 1 * n
                            else:
                                query_terms[term] = 1 * n
                        if '{}_{}'.format(self.word2id[queries[i][j+1]], -1) in self.bigram:
                            uniterm = self.bigram['{}_{}'.format(self.word2id[queries[i][j+1]], -1)]
                            if uniterm in query_terms:
                                query_terms[uniterm] += 1 * n
                            else:
                                query_terms[uniterm] = 1 * n
                    else:
                        continue
                    
            origin_queries.append(query_terms)
            tEnd = time.time()
            logging.info("Query processing cost %f sec" % (tEnd - tStart))
        
            _scores = [self._score(query_terms, index) for index in range(num_docs)]

            
            scores.append(np.array(_scores))
            tEnd = time.time()
            logging.info("Query scoring cost %f sec" % (tEnd - tStart))
        scores_indices = np.array(scores).argsort()

        return scores_indices,  origin_queries

    def _score(self, query, article_index):
        cdef float score, qfreq
        cdef float tfreq, TF, QF, IDF, denominator_term
        cdef int term
        cdef dict frequencies
        score = 0.0

        frequencies = self.tf[article_index] # dict of terms in doc
        denominator_term = self.denominator_term[article_index]
        if len(frequencies) > len(query):
            for term in query:
                if term not in frequencies:
                    continue
                qfreq = query[term]
                tfreq = frequencies[term]
                
                TF = tfreq * (k + 1) / (tfreq + denominator_term)
                QF = qfreq * (k_q+1) / (qfreq + k_q)
                IDF = self.idf[term]
                score += IDF * TF * QF
        else:
            for term in frequencies:
                if term not in query:
                    continue
                qfreq = query[term]
                tfreq = frequencies[term]
                
                IDF = self.idf[term]
                TF = tfreq * (k + 1) / (tfreq + denominator_term)
                QF = qfreq * (k_q+1) / (qfreq + k_q)
                score += IDF * TF * QF

        return score

    
    def rocchio_feedback(self, origin_queries, drs, dnrs):
        cdef float v, tf
        cdef int i, doc_id, index
        cdef int rel_term, irrel_term
        cdef dict new_query, frequencies
        cdef float num_dr
        cdef float num_dnr
        num_dr = drs.shape[1]
        num_dnr = dnrs.shape[1]
        
        print(drs)

        logging.info('Feedback..!') 
        scores = []
        for i in range(len(origin_queries)):
            tStart = time.time()
            new_query = origin_queries[i]
            dr = drs[i]
            dnr = dnrs[i]
            new_query.update((x , y*alpha) for x, y in new_query.items())
            for doc_id in dr:
                frequencies = self.tf[doc_id]
                for rel_term in frequencies:
                    tf = frequencies[rel_term]
                    v = tf * beta / num_dr
                    if rel_term in new_query:
                        new_query[rel_term] += v
                    else:
                        new_query[rel_term] = v
            
            for doc_id in dnr:
                frequencies = self.tf[doc_id]
                for irrel_term in frequencies:
                    tf = frequencies[irrel_term]
                    v = tf * gamma / num_dnr * -1
                    if irrel_term in new_query:
                        new_query[irrel_term] += v
                    else:
                        new_query[irrel_term] = v
            
            tEnd = time.time()
            logging.info("Query processing cost %f sec" % (tEnd - tStart))
            _scores = [self._score(new_query, index) for index in range(num_docs)]
            scores.append(np.array(_scores))
            tEnd = time.time()
            logging.info("Query feedback cost %f sec" % (tEnd - tStart))
        
        return np.array(scores).argsort()

    

    
