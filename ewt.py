import torch 
from torch.utils.data import Dataset
import numpy as np
from os.path import join
from os import listdir
"""
EWT: a Dataset class for the EWT Dependency Parsed Dataset
path: path to .npy and .txt files preprocessed files
split: whether we want a 'train', 'val' or 'test' split for the data
punctuation: whether we want the dataset including punctuation or not.
tokenizer: which tokenizer to use. Not required.
"""

class EWT(Dataset):

    def __init__(self, path, split='train', punctuation=True, tokenizer=None):
        super().__init__()
        # We look for all samples in folder
        self.sample_ids = []
        self.sample_sentences = []
        self.sample_matrices = []
        self.sample_tokens = []
        self.tokenizer = tokenizer
        fullpath = path
        matrices = "en_ewt-ud-{}_{}_.npy".format(split, "punct" if punctuation else 'nopunct')
        sentences= "en_ewt-ud-{}_{}_.txt".format(split, "punct" if punctuation else 'nopunct')
        
        # read sentences
        sentence_path = join(path, sentences)
        # read matrices
        matrix_path = join(path, matrices)
        
        sentence_file = open(sentence_path, "r")
        matrix_file = np.load(matrix_path, allow_pickle=True)
        lines = sentence_file.read().splitlines()
        sentence_file.close()
        # if available, tokenize sentences
        if self.tokenizer is not None:
            examples = tokenizer(lines, 
                                add_special_tokens=True,
                                truncation=True)['input_ids']
            examples =[torch.tensor(e, dtype=torch.long) for e in examples]
            self.sample_tokens.extend(examples)

        lines = [line.split(" ") for line in lines]
        
        for line in lines:
            self.sample_sentences.append(line)
        for m in matrix_file:
            self.sample_matrices.append(m)
        

    def __len__(self):
        return len(self.sample_sentences)

    def __getitem__(self, id):
        if self.tokenizer is None:
            return self.sample_sentences[id], self.sample_matrices[id].transpose()
        else:
            return self.sample_sentences[id], self.sample_matrices[id].transpose(), self.sample_tokens[id]
    
