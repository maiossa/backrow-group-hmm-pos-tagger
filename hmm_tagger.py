"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict

from conllu import TokenList
from process_data import get_data
from collections import Counter
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
        
        self.tag2idx: Dict[str,int] = {}
        self.idx2tag: Dict[int,str]= {}
        self.word2idx: Dict[str,int] = {}
        self.idx2word: Dict[int,str] = {}
        
        self.log_transition = None
        self.log_emission = None
        
        self.T = 0
        self.V = 0

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
        
        tags = []
        tokens = []

        for sentence in sentences:
            tags.append(START_TAG)
            tokens.append(START_TOKEN)

            for token in sentence:
                tags.append(token["upos"])
                tokens.append(token["form"])

            tags.append(END_TAG)
            tokens.append(END_TOKEN)

        token_counts = dict(Counter(tokens))
        unk_threshold = 2
        for i, token in enumerate(tokens):
            if token_counts[token] <= unk_threshold:
                tokens[i] = UNK_TOKEN
            
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

        transition_data[END_TAG] = {}
        transition_matrix = pd.DataFrame(transition_data).T
        transition_matrix = transition_matrix.fillna(0)
        
        
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
        emission_matrix = emission_matrix.fillna(0)

        tag_order = sorted(transition_matrix.index)
        tag_order.remove(START_TAG)
        tag_order.remove(END_TAG)
        tag_order += [START_TAG, END_TAG]

        transition_matrix = transition_matrix.reindex(index=tag_order, columns=tag_order)
        emission_matrix = emission_matrix.reindex(index=tag_order)
        n_tags = len(tag_order)
        n_words = len(emission_matrix.columns)
        
        # Smooth transition matrix
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix_smooth = transition_matrix.add(self.alpha).div(row_sums + self.alpha * n_tags, axis=0)
        
        # Smooth emission matrix
        col_sums = emission_matrix.sum(axis=0)
        emission_matrix_smooth = emission_matrix.add(self.alpha).div(col_sums + self.alpha * n_tags, axis=1)

        self.transition_df = transition_matrix_smooth
        self.emission_df = emission_matrix_smooth

        self.transition_matrix = transition_matrix_smooth.to_numpy()
        self.emission_matrix = emission_matrix_smooth.to_numpy()

        self.log_transition = np.log(self.transition_matrix)
        self.log_emission = np.log(self.emission_matrix)

        self.tag2idx = { tag: i for i, tag in enumerate(transition_matrix.index)}
        self.idx2tag = list(transition_matrix.index)

        self.word2idx = {tag: i for i, tag in enumerate(emission_matrix.columns)}
        self.idx2word = list(emission_matrix.columns)

        if not pd_return:
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

        sentence = [token if token in self.word2idx else UNK_TOKEN for token in sentence]
        return ViterbiDecoder(self).tag(sentence)


    def save_model(self, filepath: str):
        """
        saves the trained model

        Args:
            filepath (str): model path
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        np.savez(filepath,
            transition=self.transition_matrix,
            emission=self.emission_matrix,
            log_transition=self.log_transition,
            log_emission=self.log_emission,
            tag2idx=self.tag2idx,
            idx2tag=self.idx2tag,
            word2idx=self.word2idx,
            idx2word=self.idx2word)


    def load_model(self, filepath: str):
        """
        loads saved model

        Args:
            filepath (str): model path
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath, allow_pickle=True)
        self.transition_matrix = data["transition"]
        self.emission_matrix = data["emission"]
        self.log_transition = data["log_transition"]
        self.log_emission = data["log_emission"]
        self.tag2idx = data["tag2idx"].item()
        self.idx2tag = data["idx2tag"].tolist()
        self.word2idx = data["word2idx"].item()
        self.idx2word = data["idx2word"].tolist()
        return self

if __name__ == '__main__':

    sentences = get_data("data/english/gum/train.conllu")

    tagger = HMMTagger()

    print(tagger.train(sentences))
