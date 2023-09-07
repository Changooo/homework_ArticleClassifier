"""Skeleton Code for PA1"""

# -*- coding: utf-8 -*-

""" Data Load & Prepocessing Classes """

#   *** Do not import any library except already imported libraries ***
import argparse
import json
import nltk
import numpy as np
import math
import random

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pathlib import Path
from sklearn.svm import *
from tqdm import tqdm, trange
from typing import Union, List, Dict, Tuple, Optional

nltk.download('punkt')                      
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
#   *** Do not import any library except already imported libraries ***

class AI_util:
    def __init__(self):
         #   *** Do not modify the code ***
        self.extract_specific_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # Noun, Verb POS tags
        self.stop_words = set(stopwords.words('english'))
        self.word2idx: Dict[str, int]
        self.train_tfidf: List[Tuple[str, List[float], int]]
        self.test_tfidf: List[Tuple[str, List[float], int]]
        self.label2idx = {label: idx for idx, label in enumerate(['medical', 'business', 'graphics', 'food', 'entertainment', 'space', 'politics', 'sport', 'historical', 'technologie'])}
         #   *** Do not modify the code ***

    def load_data(self, data_path: Path, data_type: str = 'train') -> List[Tuple[str, List[str], int]]:
         #   *** Do not modify the code ***
        """
            Load data file & Preprocess the text data & (if train data) Make word2idx dictionary
            1. Load json data file 

            2. Preprocess the data paragraph
                1) Change the text to lowercase 2) Tokenize the paragraph 3) POS tagging 
                4) Only extract tokens with noun, verb POS tags  (use *self.extract_specific_tags)
                5) Remove the stopwords (use *self.stop_words)

                Ex) Data paragraph: 'Hello, world is so wonderful!' --> Preprocessed (Tokenized) paragraph: List['hello', 'world']

            3. (if train data) Make word to index dictionary (use *self.word2idx)
                Will be used to calculate the IDF score             
        """
    #   *** Do not modify the code below ***
        tokenized_data = list()
        if data_type == 'train':
            tokens_set = set()

        # 1. Load json data file 
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for d in tqdm(data, desc='Tokenizing... : '):
            paragraph = d['paragraph']
            tokenized_paragraph = None
    #   *** Do not modify the code above ***

        ##### EDIT HERE #####
        # 2. Preprocess the data paragraph
        # POS tagging
            tokenized_paragraph = list() 
            lower_paragraph = paragraph.lower()
            splitted_paragraph = word_tokenize(lower_paragraph)
            pos_paragraph = nltk.pos_tag(splitted_paragraph)
            for tokens in pos_paragraph:
                if tokens[1] in self.extract_specific_tags and tokens[0] not in self.stop_words:
                    tokenized_paragraph.append(tokens[0])
        ##### END #####

    #   *** Do not modify the code below ***
            if data_type == 'train':
                tokens_set.update(tokenized_paragraph)
            tokenized_data.append((d['id'], tokenized_paragraph, self.label2idx[d['label']]))

        # 3. (if train data) Make word to index dictionary (use *self.word2idx)
        if data_type == 'train':
            self.word2idx = {word: idx for idx, word, in enumerate(sorted(tokens_set))}

        return tokenized_data[:]
    #   *** Do not modify the code above ***

    def Calculate_TF_IDF_Normalization(self):
         #   *** Do not modify the code ***
        return

    def save_preprocess_result(self, std_name: Optional[str] = None, std_id: Optional[str] = None, paragraph_id: Optional[int] = None) -> None:
         #   *** Do not modify the code ***
        output_path = Path('./pa1_preprocess_{}_{}.txt'.format(std_name, std_id))
        with open(output_path, mode='w', encoding="utf-8") as f:
            f.write('Train Data Length : {} | TF-IDF Length : {}\n'.format(len(self.train_tfidf), len(self.train_tfidf[0][1])))
            f.write('Test Data Length : {} | TF-IDF Length : {}\n'.format(len(self.test_tfidf), len(self.test_tfidf[0][1])))
            if paragraph_id is not None:
                if paragraph_id < 801:
                    data = self.train_tfidf[paragraph_id-1]
                else:
                    data = self.test_tfidf[paragraph_id-len(self.train_tfidf)-1]
                normalized_tf_idf = [str(value) for value in data[1] if not value == 0]
                f.write("{}\t{}\t{}\n".format(data[0], '\t'.join(normalized_tf_idf), data[2]))
         #   *** Do not modify the code ***


def save_test_result(result: Dict[str, Union[Tuple[float, float, float, int], Tuple[float, int]]], 
                    std_name: Optional[str] = None, 
                    std_id: Optional[str] = None):
    #   *** Do not modify the code ***
    output_path = Path('./pa1_test_{}_{}.txt'.format(std_name, std_id))
    with open(output_path, mode='w', encoding="utf-8") as f:
        for model_type in ['knn', 'svm', 'mlp']:
            tmp = result[model_type]
            label_name = ['medical', 'business', 'graphics', 'food', 'entertainment', 'space', 'politics', 'sport', 'historical', 'technologie']
            
            headers = ["precision", "recall", "f1-score", "# docs"]
            name_width = max(len(cn) for cn in label_name)
            width = max(name_width, len('micro avg'))
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width) + '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9}\n'
            for label in label_name:
                report += row_fmt.format(
                    label, 
                    tmp[label][0], 
                    tmp[label][1], 
                    tmp[label][2], 
                    tmp[label][3], 
                    width=width
                )
            row_fmt_accuracy = '{:>{width}s} ' + \
                                ' {:>9.2}' * 2 + ' {:>9.2f}' + \
                                ' {:>9}\n'
            report += row_fmt_accuracy.format(
                'accuracy', 
                '', 
                '', 
                tmp['accuracy'][0], 
                tmp['accuracy'][1], 
                width=width
            )

            f.write('Model Type : {}\n'.format(model_type))
            f.write(report + '\n')