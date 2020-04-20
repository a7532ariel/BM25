import os
import re
import math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import logging
import time
import argparse
from tqdm import tqdm

num_dr = 1
num_dnr = 0

from vsmmodel import Retriever

if __name__ == "__main__":

    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser()
    
#     --query_file $query_file --out_file $out_file --model_dir $model_dir --corpus_dit $corpus_dir --feedback $feedback
    
    parser.add_argument("--query_file")
    parser.add_argument("--out_file")
    parser.add_argument("--model_dir")
    parser.add_argument("--corpus_dir")
    parser.add_argument("-feedback", action='store_true', default=False)
    
    
    args = parser.parse_args()
    
    r = Retriever(args)

    feedback = args.feedback

    if feedback:
        scores_argsorted, origin_queries = r.retrieve()
        irrelevant = scores_argsorted[:, :num_dnr] # 小到大
        relevant = np.flip(scores_argsorted[:, -num_dr:], axis=1)
        scores_argsorted = r.rocchio_feedback(origin_queries, relevant, irrelevant)

    else:
        scores_argsorted, origin_queries = r.retrieve()

    top100 = np.flip(scores_argsorted[:, -100:], axis=1)
    
    Top100 = []
    for array in top100:
        arr = []
        for indice in array:
            arr.append(r.id2filename[indice])
        Top100.append(' '.join(arr))

    root = ET.parse(args.query_file).getroot()
    index = []
    for query_node in root:
        index.append(query_node.findall('.//number')[0].text.replace('\n', '')[-3:])

    df = pd.DataFrame({'query_id': index, 'retrieved_docs': Top100})
    df.to_csv(args.out_file, index=False, sep=',')
    logging.info('DONE!')
    
