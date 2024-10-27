import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from collections import Counter
from tempfile import TemporaryDirectory

from recommenders.datasets.ebnerd import (download_mind,
                                     extract_mind,
                                     download_and_extract_glove,
                                     load_glove_matrix,
                                     word_tokenize
                                    )
from recommenders.datasets.download_utils import unzip_file
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))


# word_embedding_dim should be in [50, 100, 200, 300]
word_embedding_dim = 300

tmpdir = TemporaryDirectory()
data_path = 'data/EB-NeRD'
output_path = os.path.join(data_path, 'utils')
os.makedirs(output_path, exist_ok=True)


news = pd.read_table(os.path.join(data_path, 'ebnerd_small_train', 'news.tsv'),
                     names=['newid', 'vertical', 'subvertical', 'title',
                            'abstract', 'url', 'entities in title', 'entities in abstract'],
                     usecols = ['vertical', 'subvertical', 'title', 'abstract'])

news_vertical = news.vertical.drop_duplicates().reset_index(drop=True)
vert_dict_inv = news_vertical.to_dict()
vert_dict = {v: k+1 for k, v in vert_dict_inv.items()}

news_subvertical = news.subvertical.drop_duplicates().reset_index(drop=True)
subvert_dict_inv = news_subvertical.to_dict()
subvert_dict = {v: k+1 for k, v in vert_dict_inv.items()}

news.title = news.title.apply(word_tokenize)
news.abstract = news.abstract.apply(word_tokenize)

word_cnt = Counter()
word_cnt_all = Counter()

for i in tqdm(range(len(news))):
    word_cnt.update(news.loc[i]['title'])
    word_cnt_all.update(news.loc[i]['title'])
    word_cnt_all.update(news.loc[i]['abstract'])
  
word_dict = {k: v+1 for k, v in zip(word_cnt, range(len(word_cnt)))}
word_dict_all = {k: v+1 for k, v in zip(word_cnt_all, range(len(word_cnt_all)))}

with open(os.path.join(output_path, 'vert_dict.pkl'), 'wb') as f:
    pickle.dump(vert_dict, f)
    
with open(os.path.join(output_path, 'subvert_dict.pkl'), 'wb') as f:
    pickle.dump(subvert_dict, f)

with open(os.path.join(output_path, 'word_dict.pkl'), 'wb') as f:
    pickle.dump(word_dict, f)
    
with open(os.path.join(output_path, 'word_dict_all.pkl'), 'wb') as f:
    pickle.dump(word_dict_all, f)

glove_path = '/Resource/yangbo_storage/projects/pretrained/word2vec'

embedding_matrix, exist_word = load_glove_matrix(glove_path, word_dict, word_embedding_dim)
embedding_all_matrix, exist_all_word = load_glove_matrix(glove_path, word_dict_all, word_embedding_dim)

np.save(os.path.join(output_path, 'embedding.npy'), embedding_matrix)
np.save(os.path.join(output_path, 'embedding_all.npy'), embedding_all_matrix)

uid2index = {}

with open(os.path.join(data_path, 'ebnerd_small_train', 'behaviors.tsv'), 'r') as f:
    for l in tqdm(f):
        uid = l.strip('\n').split('\t')[1]
        if uid not in uid2index:
            uid2index[uid] = len(uid2index) + 1

with open(os.path.join(output_path, 'uid2index.pkl'), 'wb') as f:
    pickle.dump(uid2index, f)

utils_state = {
    'vert_num': len(vert_dict),
    'subvert_num': len(subvert_dict),
    'word_num': len(word_dict),
    'word_num_all': len(word_dict_all),
    'embedding_exist_num': len(exist_word),
    'embedding_exist_num_all': len(exist_all_word),
    'uid2index': len(uid2index)
}
utils_state


# Record results for tests - ignore this cell
print("vert_num", len(vert_dict))
print("subvert_num", len(subvert_dict))
print("word_num", len(word_dict))
print("word_num_all", len(word_dict_all))
print("embedding_exist_num", len(exist_word))
print("embedding_exist_num_all", len(exist_all_word))
print("uid2index", len(uid2index))
