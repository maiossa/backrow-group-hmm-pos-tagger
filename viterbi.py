"""
This module implements the Viterbi algorithm
"""
import numpy as np
from typing import List, Tuple
from hmm_tagger import START_TAG, END_TAG, UNK_TOKEN

class ViterbiDecoder:
  def __init__(self, hmm_tagger):

    self.tagger = hmm_tagger
    self.tag2idx = hmm_tagger.tag2idx
    self.idx2tag = hmm_tagger.idx2tag
    self.word2idx = hmm_tagger.word2idx
    self.idx2word = hmm_tagger.idx2word

    # convert probability matrices to log space for numerical stability
    self._prepare_log_probabilities()
  
  def _prepare_log_probabilities(self):
    epsilon = 1e-10

    # convert to log probabilities
    self.log_transition = np.log(self.tagger.transition_matrix + epsilon)
    self.log_emission = np.log(self.tagger.emission_matrix + epsilon)

  def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
    if not sentence:
      return []
    
    #get word indices
    word_indices = []
    for word in sentence:
      if word in self.word2idx:
        word_indices.append(self.word2idx[word])
      else:
        # this should happen if preprocessing was done correctly
        word_indices.append(self.word2idx.get(UNK_TOKEN, 0))
    
    #run viterbi algorithm
    tag_indices = self._viterbi(word_indices)

    #convert indices back to tags
    tags = [self.idx2tag[idx] for idx in tag_indices]

    #return (word, tag) pairs
    return list(zip(sentence, tags))
  
  def _viterbi(self, word_indices: List[int]) -> List[int]:
    n_words = len(word_indices)
    n_tags = len(self.idx2tag)

    #get indices for START and END tags
    start_idx = self.tag2idx[START_TAG]
    end_idx = self.tag2idx[END_TAG]

    #filter out START and END tags from possible tags
    valid_tag_indices = [i for i in range(n_tags) if i != start_idx and i != end_idx]
    n_valid_tags = len(valid_tag_indices)

    #initialize viterbi matrix 
    viterbi = np.full((n_words, n_valid_tags), -np.inf)

    #initialize backpointer matrix
    backpointer = np.zeros((n_words, n_valid_tags), dtype=int)

    #initialization step P(tag_0) = P(tag_0 | START) * P(word_0 | tag_0)
    word_idx_0 = word_indices[0]
    for s_idx, tag_idx in enumerate(valid_tag_indices):
      trans_prob = self.log_transition[start_idx, tag_idx]

      if word_idx_0 < self.log_emission.shape[1]:
        emit_prob = self.log_emission[tag_idx, word_idx_0]
      else:
        #word not in vocabulary
        emit_prob = -np.inf

      viterbi[0, s_idx] = trans_prob + emit_prob

    #recursion step - for each word, compute best path to each state
    for t in range(1, n_words):
      word_idx_t = word_indices[t]

      for s_idx, curr_tag_idx in enumerate(valid_tag_indices):
        #find best previous state
        max_prob = -np.inf
        best_prev_state = 0

        for prev_s_idx, prev_tag_idx in enumerate(valid_tag_indices):
          trans_prob = self.log_transition[prev_tag_idx, curr_tag_idx]

          path_prob = viterbi[t-1, prev_s_idx]

          total_prob = path_prob + trans_prob

          if total_prob > max_prob:
            max_prob = total_prob
            best_prev_state = prev_s_idx

        #Log P(word | tag)
        if word_idx_t < self.log_emission.shape[1]:
          emit_prob = self.log_emission[curr_tag_idx, word_idx_t]
        else:
          emit_prob = -np.inf

        viterbi[t, s_idx] = max_prob + emit_prob
        backpointer[t, s_idx] = best_prev_state

    #termination step - find best final state
    final_probs = np.full(n_valid_tags, -np.inf)
    for s_idx, tag_idx in enumerate(valid_tag_indices):
      #add transition probability to END tag
      trans_to_end = self.log_transition[tag_idx, end_idx]
      final_probs[s_idx] = viterbi[n_words-1, s_idx] + trans_to_end

    best_final_state = np.argmax(final_probs)

    #backtracking to find best path
    best_path_indices = [0] * n_words
    best_path_indices[n_words-1] = best_final_state

    for t in range(n_words-1, 0, -1):
      best_path_indices[t-1] = backpointer[t, best_path_indices[t]]

    #convert back to original tag indices
    best_path_tag_indices = [valid_tag_indices[idx] for idx in best_path_indices]
    return best_path_tag_indices
