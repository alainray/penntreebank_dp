import torch 
from torch.utils.data import Dataset
import numpy as np
from os.path import join
from os import listdir
"""
PennDP: a Dataset class for the Penn TreeBank Dependency Parsed Dataset
path: path to treebank_3 folder
corpus_name: one of 'wsj' or 'brown' (optional, default 'wsj')
split: whether we want a 'train', 'val' or 'test' split for the data
"""

class PennDP(Dataset):

    def __init__(self, path, corpus_name='wsj', split='train', tokenizer=None):
        super().__init__()
        # We look for all samples in folder
        self.sample_ids = []
        self.sample_sentences = []
        self.sample_matrices = []
        self.sample_tokens = []
        self.tokenizer = tokenizer
        fullpath = join(path, "parsed/mrg" , corpus_name)

        """
        WSJ splits defined according to:
        https://aclweb.org/aclwiki/POS_Tagging_(State_of_the_art)
        
        """
        if corpus_name == 'wsj':
            if split == 'test':
                splits = ["{:02}".format(x) for x in range(22,24)]
            elif split == 'val':
                splits = ["{:02}".format(x) for x in range(19,22)]
            elif split == 'train':
                splits = ["{:02}".format(x) for x in range(19)]
            else:
                splits = ["{:02}".format(x) for x in range(25)] # all of them
        
        if corpus_name == 'brown':
            if split == 'test':
                splits = ['cf','cg','ck','cl','cm']
            elif split == 'val':
                splits = ['cn']
            elif split == 'train':
                splits = ['cp','cr']
            else:
                splits = ['cf','cg','ck','cl','cm','cn','cp','cr']

        for subdir in listdir(fullpath):
            subdirpath = join(fullpath,subdir)
            if ".LOG" not in subdir and "r" != subdir and subdir in splits:
                for sample in listdir(subdirpath):
                    self.sample_ids.append(join(subdir,sample.split(".")[0]))
        
        datapath = join(path, "parsed/pd" , corpus_name)
        for path in self.sample_ids:
            # read sentences
            sentence_path = join(datapath,path + ".txt")
            # read matrices
            matrix_path = join(datapath,path + ".npy")
            
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
      
