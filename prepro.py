'''
There is no intended bug in this module.
Please do not modify this code.
'''

import os
import numpy as np
from tqdm import tqdm

from data_utils import *


prepro_filename = 'data/prepro.npz'


def read_tsv():
    datasets = {}
    for split in ['train', 'test']:
        x, y = [], []

        if not os.path.exists('data'):
            os.makedirs('data')
        with open(f'data/{split}.tsv', 'r') as f:
            dataset = csv.reader(f, delimiter='\t')
            next(dataset) # skip header: [id, document, label]

            print(f'preprocessing {split}')
            for _id, text, label in tqdm(dataset):
                x += [text2ind(text)]   # sentence preprocessing is done here
                y += [int(label)]   # label is an integer: either 0 or 1.
                sys.stdout.flush()

        datasets.update({f'x_{split}': np.asarray(x), 
                     f'y_{split}': np.asarray(y)})
    return datasets # this dictionary gets four keys: 'x_train', 'y_train', 'x_test', 'y_test'


# we use Naver Sentiment Movie Corpus v1.0
# from Lucy Park's [nsmc](https://github.com/e9t/nsmc)
# which is a dataset for binary sentiment classification of movie reviews.
def prepro_datasets():
    datasets = read_tsv()
    # save the file as 'data/prepro.npz'
    # you will use this file to train and test your network.
    np.savez(open(prepro_filename, 'wb'),
        x_train=datasets['x_train'],
        y_train=datasets['y_train'],
        x_test=datasets['x_test'],
        y_test=datasets['y_test'])


'''
Run python prepro.py to preprocess datasets as an npz file.
The file will be saved as data/npz
'''
if __name__ == '__main__':
    prepro_datasets()
    print(f'saved {prepro_filename}')