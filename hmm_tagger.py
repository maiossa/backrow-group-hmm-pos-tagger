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

        self.transition_matrix = None
        self.emission_matrix = None

    def train(self, training_data: list[TokenList], pd_return=False):
        """
        train the HMM and fill emission and transition matrixes

        args:
            training_data (list[TokenList]): Parsed sentences.
                Each token in a TokenList provides:
                - token["form"] → surface word
                - token["upos"] → universal POS tag
                These are used to build transition and emission counts.

            pd_return (Bool): return matrixes in a human readable pandas dataset format
        
        returns:
            tuple[np.ndarray, np.ndarray]:
                - transition_matrix: P(tag_next | tag_current)
                - emission_matrix:  P(word | tag)
        """

        sentences = training_data
        
        # flatten sentences into lists of tokens and tags
        tags = []
        tokens = []

        for sentence in sentences:

            tags.append("START")
            tokens.append("START_TOKEN")

            for token in sentence:
                # upos: Universal POS tags
                tags.append(token["upos"])
                # form: the word
                tokens.append(token["form"])

            tags.append("END")
            tokens.append("END_TOKEN")

        # Define the transition matrix ######################
            
        transition_counts = {}

        for i in range(len(tags) - 1):
            current_tag = tags[i]
            next_tag = tags[i + 1]

            if current_tag not in transition_counts:
                transition_counts[current_tag] = []

            transition_counts[current_tag].append(next_tag)

        # normalize counts to create a normaized distribution
        transition_data = {}
        for tag, next_tags in transition_counts.items():
            tag_counter = Counter(next_tags)
            prob_dist = {k: v / len(next_tags) for k, v in tag_counter.items()}
            transition_data[tag] = prob_dist

        # making sure an end token stays the ending token.
        transition_data["END"] = {}
    
        transition_matrix = pd.DataFrame(transition_data).T
        # Take this pandas version for a more human-readeable output:
        transition_matrix = transition_matrix.fillna(0) 

        if not pd_return:
            self.tag2idx = { tag: i for i, tag in enumerate(transition_matrix.index)}
            self.idx2tag = list(transition_matrix.index)
            # But this should be faster when it comes to processing:
            transition_matrix = transition_matrix.to_numpy()


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
        if not pd_return:
            self.word2idx = {tag: i for i, tag in enumerate(emission_matrix.columns)}
            self.idx2word = list(emission_matrix.columns)
            emission_matrix = emission_matrix.to_numpy() # But this should be faster when it comes to processing

        # store matrixes
        self.transition_matrix, self.emission_matrix = transition_matrix, emission_matrix

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

        # ensure correct file extension
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        np.savez(filepath,
            transition=self.transition_matrix,
            emission=self.emission_matrix)


    def load_model(self, filepath: str):
        """
        loads saved model

        Args:
            filepath (str): model path
        """

        # ensure correct file extension
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath, allow_pickle=True)
        self.transition_matrix = data["transition"]
        self.emission_matrix = data["emission"]
        return self

##################################

if __name__ == '__main__':

    sentences = get_data("data/english/gum/train.conllu")

    tagger = HMMTagger()

    print(tagger.train(sentences))
