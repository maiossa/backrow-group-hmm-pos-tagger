from .tagger_helpers import (
    normalize_input,
    pair_words_with_tags,
)
from .viterbi import (
    viterbi_forward,
    viterbi_backtrack
)


def viterbi_tag(hmm, sentence):
    # normalize + add START/END
    sentence = normalize_input(sentence, hmm.word2idx)
    n_tokens = len(sentence)

    # forward pass
    viterbi, backpointer = viterbi_forward(
        sentence,
        hmm.T,
        hmm.tag2idx,
        hmm.word2idx,
        hmm.transition_matrix,
        hmm.emission_matrix,
    )

    # backward pass
    predicted_tags = viterbi_backtrack(
        backpointer,
        hmm.tag2idx,
        hmm.idx2tag,
        n_tokens,
    )

    # pair (word, tag)
    return pair_words_with_tags(sentence, predicted_tags)
