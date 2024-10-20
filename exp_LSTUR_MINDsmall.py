import os
import sys
import numpy as np
import zipfile
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

mode = 'eval'  # train / eval

epochs = 5
seed = 40
batch_size = 64

# Options: demo, small, large
MIND_type = "small"
data_path = './data'

train_news_file = os.path.join(data_path, 'MINDsmall_train_from_train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'MINDsmall_train_from_train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'MINDsmall_val_from_train', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'MINDsmall_val_from_train', r'behaviors.tsv')
test_news_file = os.path.join(data_path, 'MINDsmall_dev', r'news.tsv')
test_behaviors_file = os.path.join(data_path, 'MINDsmall_dev', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print(hparams)

iterator = MINDIterator
model = LSTURModel(hparams, iterator, seed=seed)

if mode == 'train':
  model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
  model_path = os.path.join("model/LSTUR_MINDsmall_1")
  os.makedirs(model_path, exist_ok=True)
  model.model.save_weights(os.path.join(model_path, "lstur_ckpt"))
else:
  checkpoint_path = os.path.join("model/LSTUR_MINDsmall_1", "lstur_ckpt")
  model.model.load_weights(checkpoint_path)

res_syn = model.run_eval(test_news_file, test_behaviors_file)
print(res_syn)

# Record results for tests - ignore this cell
store_metadata("group_auc", res_syn['group_auc'])
store_metadata("mean_mrr", res_syn['mean_mrr'])
store_metadata("ndcg@5", res_syn['ndcg@5'])
store_metadata("ndcg@10", res_syn['ndcg@10'])
