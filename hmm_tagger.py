# src/hmm_tagger.py
"""
hmm_tagger.py

This module implements an HMM based POS-tagging-system.
Currently only a skeleton.
"""

from typing import List, Tuple, Dict


class HMMTagger:
    """
    HHM POS-tagger
    """

    def __init__(self):
        """
        initializes the HMM
        """
        pass

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """
        trains HHM with labeled sentences

        Args:
            tagged_sentences (List[List[Tuple[str, str]]]): list of sentences. every sentence is a list of (word, tag) tuples.
        """
        pass

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
