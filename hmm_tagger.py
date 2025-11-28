"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict

from conllu import TokenList
from process_data import get_data
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

START_TAG = "START"
START_TOKEN = "START_TOKEN"
END_TAG = "END"
END_TOKEN = "END_TOKEN"
UNK_TOKEN = "UNK"

class HMMTagger:

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        print("hell1o")
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

        self.transition_matrix: np.ndarray = None
        self.emission_matrix: np.ndarray = None

        self.transition_df: pd.DataFrame = None
        self.emission_df: pd.DataFrame = None


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

            tags.append(START_TAG)
            tokens.append(START_TOKEN)

            for token in sentence:
                # upos: Universal POS tags
                tags.append(token["upos"])
                # form: the word
                tokens.append(token["form"])

            tags.append(END_TAG)
            tokens.append(END_TOKEN)


        # replace less frequent words with unk tokens
        token_counts = dict(Counter(tokens))
        unk_threshold = 2
        for i, token in enumerate(tokens):
            if token_counts[token] <= unk_threshold:
                tokens[i] = UNK_TOKEN

        # Define the transition matrix ######################
            
        transition_counts = {}

        for i in range(len(tags) - 1):
            current_tag = tags[i]
            next_tag = tags[i + 1]

            if current_tag not in transition_counts:
                transition_counts[current_tag] = []

            transition_counts[current_tag].append(next_tag)

        #  compute transition probabilities using Laplace smoothing
        transition_data = {}
        all_tags = sorted(set(tags))

        for tag in all_tags:
            transition_data[tag] = {}
            next_tags = transition_counts.get(tag, [])
           # how many times this tag appeared as previous
            total_count = len(next_tags)

            tag_counter = Counter(next_tags)

            for nxt in all_tags:
                raw = tag_counter[nxt]
                smoothed = (raw + self.alpha) / (total_count + self.alpha * len(all_tags))
                transition_data[tag][nxt] = smoothed

        # making sure an end token stays the ending token.
        transition_data[END_TAG] = {}
    
        # Take this pandas version for a more human-readeable output:
        transition_matrix = pd.DataFrame(transition_data).T
        transition_matrix = transition_matrix.fillna(0) 


        # Define the emission matrix ######################
        
        
        emission_counts = {}

        for i in range(len(tokens)):
            current_tag = tags[i]
            current_token = tokens[i]

            if current_token not in emission_counts:
                emission_counts[current_token] = []

            emission_counts[current_token].append(current_tag)

        #  compute emission probabilities using Laplace smoothing
        emission_data = {}
        vocab = sorted(set(tokens))

        for token, tag_list in emission_counts.items():

            tag_counter = Counter(tag_list)
            total_count = len(tag_list)

            for tag in all_tags:
                raw = tag_counter[tag]
                smoothed = (raw + self.alpha) / (total_count + self.alpha * len(vocab))
                if tag not in emission_data:
                    emission_data[tag] = {}
                emission_data[tag][token] = smoothed

        emission_matrix = pd.DataFrame(emission_data)
        emission_matrix = emission_matrix.fillna(0) # Take this version for a more human-readeable output
        

        # tag order to be used to sort dataframe columns and rows
        tag_order = sorted(transition_matrix.index)
        tag_order.remove(START_TAG)
        tag_order.remove(END_TAG)
        tag_order += [START_TAG, END_TAG]

        # reorder tags
        transition_matrix = transition_matrix.reindex(index=tag_order, columns=tag_order)
        emission_matrix = emission_matrix.reindex(index=tag_order)

        self.transition_df = transition_matrix
        self.emission_df = emission_matrix

        self.transition_matrix = transition_matrix.to_numpy()
        self.emission_matrix = emission_matrix.to_numpy()

        self.tag2idx = { tag: i for i, tag in enumerate(transition_matrix.index)}
        self.idx2tag = list(transition_matrix.index)

        self.word2idx = {tag: i for i, tag in enumerate(emission_matrix.columns)}
        self.idx2word = list(emission_matrix.columns)


        if not pd_return:
            # return numpy matrices
            return self.transition_matrix, self.emission_matrix
   
        return transition_matrix, emission_matrix
    

    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        taggs a sentence of words with POS-tags

        Args:
            sentence (List[str]): list of words

        Returns:
            List[Tuple[str, str]]: list of (words, tag) tuples
        """

        from viterbi import ViterbiDecoder

        # replace unseen words with unk
        sentence = {token if token in self.word2idx else UNK_TOKEN for token in sentence}

        return ViterbiDecoder(self).tag(sentence)


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
