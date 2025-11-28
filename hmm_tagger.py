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
            if token_counts[tokens] <= unk_threshold:
                tokens[i] = UNK_TOKEN

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

        emission_data = {}
        for token, tags in emission_counts.items():
            tag_counter = Counter(tags)
            prob_dist = {k: v / len(tags) for k, v in tag_counter.items()}
            emission_data[token] = prob_dist

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

        ####################################
        # <PSEUDOCODE>
        ####################################

        # Add a START_TOKEN (with tag) to the beginning of the input
        # Add a END_TOKEN (with tag) to the end of the input

        # Create a new table which should have:
            # one column for each token in the sequence
            # one row for each possible tag
            # START and END tokens should be already tagged with a 100% certainty

        # For every column (Except the START and END TOKENS)
            # For every cell in that column

                # Create an empty dictionary for the probabilities

                # For i in every possible tag

                    # Get the probability of the current row tag being the correct one, given that the previous tag is i (That is, the probability of the previous tag being i times the probability of this row tag following i)

                    # Get the probability of the current row tag being the correct one, given the token.
                    # Multiply them and save that into the dictionary

                # Take the highest probability in the dictionary and save that for this cell. This is the probability of the row tag being correct


        # Once the table is finished pick the best probability in each column. That row is the predicted tag for each token. 

        ####################################
        # <\PSEUDOCODE>
        ####################################
        return ViterbiDecoder(self).tag(sentence)


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

    print(tagger.train(sentences))
