"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict
from process_data import get_data
from collections import Counter
import numpy as np
import pandas as pd

class HMMTagger:
  
    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        
        # Mappings
        self.tag2idx: Dict[str,int] = {}
        self.idx2tag: Dict[int,str]= {}
        self.word2idx: Dict[str,int] = {}
        self.idx2word: Dict[int,str] = {}
        
        #Probability matrices
        self.log_transition = None
        self.log_emission = None
        
        #Sizes
        self.T = 0 #tag count
        self.V = 0 #vocabulary size (word count)
        
        self.start_tag = "<START>"
        self.unk_word = "<UNK>"

    def train(self, training_data):

        sentences = training_data
        
       # Define the transition matrix

       ## Get the sequence of tags

        tags = []

        for sentence in sentences:

            tags.append("START_TOKEN")

            for token in sentence:
        
                tags.append(token["upos"])

            tags.append("END_TOKEN")

        ## Calculate the counts

        post_ADJ = []
        post_ADP = []
        post_ADV = []
        post_AUX = []
        post_CCONJ = []
        post_DET = []
        post_INTJ = []
        post_NOUN = []
        post_PART = []
        post_PRON = []
        post_PROPN = []
        post_PUNCT = []
        post_SCONJ = []
        post_SYM = []
        post_VERB = []
        post_X = []
        post_START_TOKEN = []

        index = -1

        for tag in tags:
            
            index += 1

            if tag == "ADJ":
                post_ADJ.append(tags[index + 1])
            if tag == "ADP":
                post_ADP.append(tags[index + 1])
            if tag == "ADV":
                post_ADV.append(tags[index + 1])               
            if tag == "AUX":
                post_AUX.append(tags[index + 1])
            if tag == "CCONJ":
                post_CCONJ.append(tags[index + 1])
            if tag == "DET":
                post_DET.append(tags[index + 1])
            if tag == "INTJ":
                post_INTJ.append(tags[index + 1])
            if tag == "NOUN":
                post_NOUN.append(tags[index + 1])
            if tag == "PART":
                post_PART.append(tags[index + 1])
            if tag == "PRON":
                post_PRON.append(tags[index + 1])
            if tag == "PROPN":
                post_PROPN.append(tags[index + 1])
            if tag == "PUNCT":
                post_PUNCT.append(tags[index + 1])
            if tag == "SCONJ":
                post_SCONJ.append(tags[index + 1])
            if tag == "SYM":
                post_SYM.append(tags[index + 1])
            if tag == "VERB":
                post_VERB.append(tags[index + 1])
            if tag == "X":
                post_X.append(tags[index + 1])
            if tag == "START_TOKEN":
                post_START_TOKEN.append(tags[index + 1])

        def get_prob_dist(variable):
            prob_dist = Counter(variable)

            for key, value in prob_dist.items():
                prob_dist[key] = value / len(variable)

            return prob_dist

        # Define transform the counts into probability distributuons
        transition_data = {
                "post_ADJ": get_prob_dist(post_ADJ),

                "post_ADP": get_prob_dist(post_ADP),

                "post_ADV": get_prob_dist(post_ADV),

                "post_AUX": get_prob_dist(post_AUX),

                "post_CCONJ": get_prob_dist(post_CCONJ),

                "post_DET": get_prob_dist(post_DET),

                "post_INTJ": get_prob_dist(post_INTJ),

                "post_NOUN": get_prob_dist(post_NOUN),

                "post_PART": get_prob_dist(post_PART),

                "post_PRON": get_prob_dist(post_PRON),

                "post_PROPN": get_prob_dist(post_PROPN),

                "post_PUNCT": get_prob_dist(post_PUNCT),

                "post_SCONJ": get_prob_dist(post_SCONJ),

                "post_SYM": get_prob_dist(post_SYM),

                "post_VERB": get_prob_dist(post_VERB),

                "post_X": get_prob_dist(post_X),

                "post_START_TOKEN": get_prob_dist(post_START_TOKEN),

                "post_END_TOKEN": {}  

                }

        transition_matrix = pd.DataFrame(transition_data).T

        transition_matrix = transition_matrix.fillna(0)
        
        # WHAT IS MISSING RIGHT NOW
        # calculate the emission matrix
        # the matrices should be in numpy, not pandas
        # the way I obtain the transition matrix is straight up atrocious, needs refactoring
        emission_matrix = "PLACEHOLDER"
        return transition_matrix, emission_matrix

    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        taggs a sentence of words with POS-tags

        Args:
            sentence (List[str]): list of words

        Returns:
            List[Tuple[str, str]]: list of (words, tag) tuples
        """
        pass

    def save_model(self, filepath: str):
        """
        saves the trained model

        Args:
            filepath (str): model path
        """
        pass

    def load_model(self, filepath: str):
        """
        loads saved model

        Args:
            filepath (str): model path
        """
        pass

##################################

sentences = get_data("data/english/gum/train.conllu")

tagger = HMMTagger()

print(tagger.train(sentences))
