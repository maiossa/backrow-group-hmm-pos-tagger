from .train_helpers import (
    extract_tokens_tags,
    replace_rare_tokens,
    build_transition,
    build_emission,
    reorder,
)
import numpy as np


def build_hmm_parameters(training_data, mask_rate = 0.0):
    sentences = training_data

    # Extract tags and tokens
    tags, words = extract_tokens_tags(sentences, mask_rate = mask_rate)

    # Replace rare
    words = replace_rare_tokens(words, threshold=1)

    # Build matrices
    transition_df = build_transition(tags)
    emission_df   = build_emission(tags, words)

    # Reorder
    transition_df, emission_df, tags = reorder(
        transition_df, emission_df
    )
    words = emission_df.columns

    return transition_df, emission_df, tags, words


def _assign_parameters(hmm, transition_df, emission_df, tags, words):
    # Store DataFrames
    hmm.transition_df = transition_df
    hmm.emission_df   = emission_df

    # Convert to numpy
    hmm.transition_matrix = transition_df.to_numpy()
    hmm.emission_matrix   = emission_df.to_numpy()

    # Build mappings
    hmm.tag2idx = {tag: i for i, tag in enumerate(tags)}
    hmm.idx2tag = tags

    hmm.word2idx = {word: i for i, word in enumerate(words)}
    hmm.idx2word = list(words)

    # Sizes
    hmm.T = len(hmm.tag2idx)
    hmm.V = len(hmm.word2idx)

    # Log probabilities + smoothing
    hmm.log_transition = np.log(hmm.transition_matrix + hmm.alpha)
    hmm.log_emission   = np.log(hmm.emission_matrix + hmm.alpha)


def train_hmm(hmm, training_data, pd_return=False, mask_rate = 0.0):
    transition_df, emission_df, tags, words = \
        build_hmm_parameters(training_data, mask_rate = mask_rate)

    _assign_parameters(hmm, transition_df, emission_df, tags, words)

    if pd_return:
        return transition_df, emission_df

    return hmm.transition_matrix, hmm.emission_matrix
