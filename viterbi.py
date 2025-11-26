from hmm_tagger import HMMTagger
import numpy as np
import pandas as pd
from typing import List

def to_log_space(matrix, eps=1e-12):
    matrix = np.where(matrix == 0, eps, matrix)
    return np.log(matrix)


def df_to_log_space(df, eps=1e-12):
    return pd.DataFrame(
        to_log_space(df.to_numpy(), eps), 
        index=df.index, 
        columns=df.columns
        )


class ViterbiDecoder:
    def __init__(self, tagger: HMMTagger):
        '''
        :param tagger: object containing transition and emission matrices
        '''

        self.T = to_log_space(tagger.transition_matrix)
        self.E  = to_log_space(tagger.emission_matrix)
        
        self.t = df_to_log_space(tagger.transition_df)
        self.e = df_to_log_space(tagger.emission_df)

        self.n_tags = self.T.shape[0]
        self.tags = list(self.t.columns)
        self.tagger = tagger
        
    def tag(self, text: List[str]):
        return self.greedy_tag(text)

    def greedy_tag(self, text: List[str]):
        tags = []
        for word in text:
            idx = self.e[word].argmax()
            tag = self.tags[idx]
            tags += [tag]

        return tags
    
    
    