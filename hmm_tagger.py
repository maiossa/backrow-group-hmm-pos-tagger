"""
This module implements an HMM based POS-tagging-system.
"""

from typing import List, Tuple, Dict

from conllu import TokenList
from process_data import get_data
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import re

START_TAG = "START"
START_TOKEN = "START_TOKEN"
END_TAG = "END"
END_TOKEN = "END_TOKEN"
UNK_TOKEN = "UNK"

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

        self.transition_matrix: np.ndarray = None
        self.emission_matrix: np.ndarray = None

        self.transition_df: pd.DataFrame = None
        self.emission_df: pd.DataFrame = None

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
        sentences = training_data
        
        # Flatten sentences into lists of tokens and tags
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

        # Replace less frequent words with UNK tokens
        token_counts = Counter(tokens)
        unk_threshold = 1
       
        for i, token in enumerate(tokens):
            if token_counts[token] <= unk_threshold:
                tokens[i] = UNK_TOKEN

        ####################
        # Maybe we should mask at random
        ####################

        # Build transition matrix
        transition_counts = defaultdict(list)
        
        for i in range(len(tags) - 1):
            current_tag = tags[i]
            next_tag = tags[i + 1]
            transition_counts[current_tag].append(next_tag)

        # Normalize counts to create probability distribution
        transition_data = {}
        for tag, next_tags in transition_counts.items():
            tag_counter = Counter(next_tags)
            total = len(next_tags)
            prob_dist = {k: v / total for k, v in tag_counter.items()}
            transition_data[tag] = prob_dist

        # Ensure END_TAG doesn't transition anywhere
        transition_data[END_TAG] = {}

        # Create transition DataFrame
        transition_matrix = pd.DataFrame(transition_data).T
        transition_matrix = transition_matrix.fillna(0)

        # Build emission matrix
        emission_counts = defaultdict(list)
        
        for i in range(len(tokens)):
            current_tag = tags[i]
            current_token = tokens[i]
            emission_counts[current_token].append(current_tag)

        # Normalize counts to create probability distribution
        emission_data = {}
        for token, tag_list in emission_counts.items():
            tag_counter = Counter(tag_list)
            total = len(tag_list)
            prob_dist = {k: v / total for k, v in tag_counter.items()}
            emission_data[token] = prob_dist

        emission_matrix = pd.DataFrame(emission_data)
        emission_matrix = emission_matrix.fillna(0)

        # Create consistent tag ordering
        tag_order = sorted(transition_matrix.index)
        tag_order.remove(START_TAG)
        tag_order.remove(END_TAG)
        tag_order = tag_order + [START_TAG, END_TAG]

        # Reorder matrices
        transition_matrix = transition_matrix.reindex(index=tag_order, columns=tag_order, fill_value=0)
        emission_matrix = emission_matrix.reindex(index=tag_order, fill_value=0)

        # Store DataFrames
        self.transition_df = transition_matrix
        self.emission_df = emission_matrix

        # Convert to numpy arrays
        self.transition_matrix = transition_matrix.to_numpy()
        self.emission_matrix = emission_matrix.to_numpy()

        # Build vocabulary mappings
        self.tag2idx = {tag: i for i, tag in enumerate(tag_order)}
        self.idx2tag = tag_order
        
        self.word2idx = {word: i for i, word in enumerate(emission_matrix.columns)}
        self.idx2word = list(emission_matrix.columns)
        
        # Store sizes
        self.T = len(self.tag2idx)
        self.V = len(self.word2idx)

        # Compute log probabilities with smoothing
        self.log_transition = np.log(self.transition_matrix + self.alpha)
        self.log_emission = np.log(self.emission_matrix + self.alpha)

        if pd_return:
            return transition_matrix, emission_matrix
        
        return self.transition_matrix, self.emission_matrix
       
    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        Tags a sentence of words with POS-tags using Viterbi algorithm.

        Args:
            sentence (List[str]): list of words

        Returns:
            List[Tuple[str, str]]: list of (word, tag) tuples
        """

        # If the input is a string
        if isinstance(sentence, str):
            sentence = re.findall(r'\w+|[^\w\s]', sentence)

        # If the input is in UD format
        if isinstance(sentence, TokenList):

            tokens = []

            for token in sentence:
                tokens.append(token["form"])

            sentence = tokens

        # Replace unseen words with UNK
        sentence = [token if token in self.word2idx else UNK_TOKEN for token in sentence]
        
        # Add START and END tokens
        sentence = [START_TOKEN] + sentence + [END_TOKEN]
        n_tokens = len(sentence)
        
        # Define the table
        viterbi = np.zeros((self.T, n_tokens))
        backpointer = np.zeros((self.T, n_tokens), dtype=int)
        
        start_idx = self.tag2idx[START_TAG]
        viterbi[start_idx, 0] = 1.0
        
        # Fill it in
        for t in range(1, n_tokens):
            token = sentence[t]
            token_idx = self.word2idx.get(token, self.word2idx.get(UNK_TOKEN))
            
            for curr_tag_idx in range(self.T):
                max_prob = -np.inf
                best_prev_tag = 0
                
                for prev_tag_idx in range(self.T):
                    # Transition probability: P(curr_tag | prev_tag)
                    trans_prob = self.transition_matrix[prev_tag_idx, curr_tag_idx]
                    
                    # Emission probability: P(token | curr_tag)
                    emis_prob = self.emission_matrix[curr_tag_idx, token_idx]
                    
                    # Total probability
                    prob = viterbi[prev_tag_idx, t-1] * trans_prob * emis_prob
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag_idx
                
                viterbi[curr_tag_idx, t] = max_prob
                backpointer[curr_tag_idx, t] = best_prev_tag
        
        # Backward pass: backtrack to find the best path
        best_path = []
        
        # Start from END token
        end_idx = self.tag2idx[END_TAG]
        best_path.append(end_idx)
        
        # Backtrack through the sequence
        for t in range(n_tokens - 1, 0, -1):
            prev_tag_idx = backpointer[best_path[-1], t]
            best_path.append(prev_tag_idx)
        
        # Reverse to get correct order and convert to tag names
        best_path.reverse()
        predicted_tags = [self.idx2tag[idx] for idx in best_path]
        
        # Remove START and END tokens and pair with original words
        result = []
        for i in range(1, len(sentence) - 1):  # Skip START and END
            original_word = sentence[i]
            predicted_tag = predicted_tags[i]
            result.append((original_word, predicted_tag))
        
        return result
        
    def test(self, testing_data: list[TokenList]):

        ####################################
        # <PSEUDOCODE>
        ####################################

        
        # Tag the sentences using the tag() function
        # Save the prediction

        # predicted = tag(testing_data[sentences])
        # real = testing_data[real_tags]

        # Make sure that these are in the exact same format
        # Until I have the the tag() function ready, I won't make any assumptions about what that will be
        # Therefore, this is all pseudocode for now

        # acurracy = mean(predicted == target)

        # sentence_level_accuracy = []

        # for sentence in sentences:
            # sentence_level_accuracy.append[predicted[sentence] == real[sentence]]

        # sentence_level_accuracy = mean(sentence_level_accuracy)

        # taggs_f1 = {} 
        
        # for tag in tags:
            # taggs_f1[tag] = get_f1
    
        ####################################
        # <\PSEUDOCODE>
        ####################################

        return 


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

    tagger.train(sentences)

    #print(tagger.tag("Alright, this should work now."))
    print(sentences[2])
    print(tagger.tag(sentences[2]))
