from conllu import TokenList
from typing import List, Tuple

from .train import train_hmm
from .tagger import viterbi_tag
from .eval_helpers import evaluate_hmm
from .serialization import save_hmm_params, load_hmm_params


class HMMTagger():
    
    def __init__(self, transition=None, emission=None, vocab=None, tags=None, alpha=1e-2):
        self.alpha = alpha
        
        self.transition_matrix = transition
        self.emission_matrix = emission
        
        self.word2idx = {w: i for i, w in enumerate(vocab)} if vocab else {}
        self.tag2idx  = {t: i for i, t in enumerate(tags)} if tags else {}
        self.idx2word = vocab or []
        self.idx2tag = tags or []

        self.T = len(tags) if tags else 0
        self.V = len(vocab) if vocab else 0


    @classmethod
    def from_params(cls, transition, emission, vocab, tags, alpha=1e-2):
        return cls(
            transition=transition,
            emission=emission,
            vocab=vocab,
            tags=tags,
            alpha=alpha
        )


    def train(self, training_data: list[TokenList], pd_return=False):
        """
        Train the HMM and fill emission and transition matrices.

        Args:
            training_data (list[TokenList]): Parsed sentences.
                Each token in a TokenList provides:
                - token["form"] → surface word
                - token["upos"] → universal dependencies POS tag
            pd_return (bool): Return matrices in pandas DataFrame format

        Returns:
            tuple[np.ndarray, np.ndarray] or tuple[pd.DataFrame, pd.DataFrame]:
                - transition_matrix: P(tag_next | tag_current)
                - emission_matrix: P(word | tag)
        """

        return train_hmm(self, training_data, pd_return)
    

    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        Tags a sentence of words with POS-tags using Viterbi algorithm.

        Args:
            sentence (List[str]): list of words

        Returns:
            List[Tuple[str, str]]: list of (word, tag) tuples
        """

        return viterbi_tag(self, sentence)
    

    def evaluate(self, test_sentences):
        """
        Evaluate HMM POS tagger.
        Returns: (accuracy, micro_f1, macro_f1)
        """
        
        return evaluate_hmm(self, test_sentences)


    def save_model(self, filepath: str):
        """
        saves the trained model

        Args:
            filepath (str): model path
        """

        save_hmm_params(filepath, self.transition_matrix, self.emission_matrix)


    def load_model(self, filepath: str):
        """
        loads saved model

        Args:
            filepath (str): model path
        """

        self.transition_matrix, self.emission_matrix = load_hmm_params(filepath)
        return self