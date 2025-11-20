"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict

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
