
from collections import defaultdict
import numpy as np

def word_2_vec(text_file):
    dic300 = [x.strip() for x in open(text_file, 'r')]
    dic300 = [x.split(' ') for x in dic300]
    word_dict = defaultdict(list)
    for entry in dic300:
        word_dict[entry[0]] = np.asarray(entry[1:], dtype='float32')
    return word_dict

word_dict = word_2_vec('glove.6B.300d.txt')
