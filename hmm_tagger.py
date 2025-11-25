"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict

from conllu import TokenList
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

    def train(self, training_data: list[TokenList]):
        """
        train the HMM and fill emission and transition matrixes

        args:
            training_data (list[TokenList]): Parsed sentences.
                Each token in a TokenList provides:
                - token["form"] → surface word
                - token["upos"] → universal POS tag
                These are used to build transition and emission counts.

        returns:
            tuple[np.ndarray, np.ndarray]:
                - transition_matrix: P(tag_next | tag_current)
                - emission_matrix:  P(word | tag)
        """

        sentences = training_data
        
        tags = []
        tokens = []

        for sentence in sentences:

            tags.append("START_TOKEN")
            tokens.append("START_TOKEN")

            for token in sentence:
        
                tags.append(token["upos"])
                tokens.append(token["form"])

            tags.append("END_TOKEN")
            tokens.append("END_TOKEN")

        # Define the transition matrix ######################
            
        transition_counts = {}

        for i in range(len(tags) - 1):
            current_tag = tags[i]
            next_tag = tags[i + 1]

            if current_tag not in transition_counts:
                transition_counts[current_tag] = []

            transition_counts[current_tag].append(next_tag)

        transition_data = {}
        for tag, next_tags in transition_counts.items():
            tag_counter = Counter(next_tags)
            prob_dist = {k: v / len(next_tags) for k, v in tag_counter.items()}
            transition_data[tag] = prob_dist

        transition_data["END_TOKEN"] = {}
    
        transition_matrix = pd.DataFrame(transition_data).T
        transition_matrix = transition_matrix.fillna(0) # Take this version for a more human-readeable output

        transition_matrix = transition_matrix.to_numpy # But this should be faster when it comes to processing


        # Define the emission matrix ######################
        
        emission_counts = {}

        for i in range(len(tokens)):
            current_tag = tags[i]
            current_token = tokens[i]

            if current_token not in emission_counts:
                emission_counts[current_token] = []

            emission_counts[current_token].append(current_tag)

        emission_data = {}
        for token, tags in emission_counts.items():
            tag_counter = Counter(tags)
            prob_dist = {k: v / len(tags) for k, v in tag_counter.items()}
            emission_data[token] = prob_dist

        emission_matrix = pd.DataFrame(emission_data)
        emission_matrix = emission_matrix.fillna(0) # Take this version for a more human-readeable output

        emission_matrix = emission_matrix.to_numpy # But this should be faster when it comes to processing

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
