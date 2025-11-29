import numpy as np
from .tokens import START_TAG, END_TAG, UNK_TOKEN


def viterbi_forward(
    sentence,
    T,
    tag2idx,
    word2idx,
    transition_matrix,
    emission_matrix,
):
    n_tokens = len(sentence)
    viterbi = np.zeros((T, n_tokens))
    backpointer = np.zeros((T, n_tokens), dtype=int)

    start_idx = tag2idx[START_TAG]
    viterbi[start_idx, 0] = 1.0

    # forward pass (fill the table)
    for t in range(1, n_tokens):
        token = sentence[t]
        token_idx = word2idx.get(token, word2idx.get(UNK_TOKEN))

        for curr_tag_idx in range(T):
            max_prob = -np.inf
            best_prev_tag = 0

            for prev_tag_idx in range(T):
                # Transition probability: P(curr_tag | prev_tag)
                trans_prob = transition_matrix[prev_tag_idx, curr_tag_idx]
                
                # Emission probability: P(token | curr_tag)
                emis_prob = emission_matrix[curr_tag_idx, token_idx]

                # Total probability
                prob = viterbi[prev_tag_idx, t - 1] * trans_prob * emis_prob

                # print(prob)
                # print(max_prob)

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag_idx

            viterbi[curr_tag_idx, t] = max_prob
            backpointer[curr_tag_idx, t] = best_prev_tag

    return viterbi, backpointer


def viterbi_backtrack(backpointer, tag2idx, idx2tag, n_tokens):
    # Backward pass: backtrack to find the best path
    best_path = []

    # Start from END token
    end_idx = tag2idx[END_TAG]
    best_path.append(end_idx)

    # Backtrack through the sequence
    for t in range(n_tokens - 1, 0, -1):
        prev_tag_idx = backpointer[best_path[-1], t]
        best_path.append(prev_tag_idx)

    # Reverse to get correct order and convert to tag names
    best_path.reverse()
    
    predicted = [idx2tag[idx] for idx in best_path]
    return predicted
