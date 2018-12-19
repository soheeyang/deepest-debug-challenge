'''
There is no intended bug in this module.
You do not need to modify this code.
'''

import sys
import csv
import numpy as np


PAD = '<pad>'
max_len = 140

def get_all_chars():
    koreans = [chr(i) for i in range(44032, 55204)] # 가-힣
    korean_chars = [chr(i) for i in range(ord('ㄱ'), ord('ㅣ') + 1)] # ㄱ-ㅎ, ㅏ-ㅣ
    other_chars = [chr(i) for i in range(ord('!'), ord('~') + 1)] # eng, digits, etc.
    return [PAD, ' '] + koreans + korean_chars + other_chars

# build char vocabulary
vocabs = get_all_chars()
ind2vocab = {ind: char for ind, char in enumerate(vocabs)}
vocab2ind = {char: ind for ind, char in enumerate(vocabs)}

def prepro_text(text):
    return ''.join(filter(lambda char: char in vocabs, text))

def text2ind(text, raw_text=True, max_len=max_len):
    if raw_text:
        text = prepro_text(text)
    return np.asarray(list(map(lambda char: vocab2ind[char], text))[:max_len] + \
            [vocab2ind[PAD] for _ in range(max((max_len - len(text)), 0))])

def ind2text(inds):
    return ''.join(map(lambda ind: ind2vocab[ind] if ind >= 0 else '', inds))
