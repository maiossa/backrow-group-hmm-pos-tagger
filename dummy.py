import numpy as np

def generate_dummy_hmm():
    # Tags and vocab
    tags = ["START", "NOUN", "VERB", "DET", "END"]
    vocab = ["the", "dog", "barks", "a"]

    T = len(tags)
    V = len(vocab)

    # Transition matrix T x T
    # rows = current tag, cols = next tag
    transition = np.array([
        [0,   0.5, 0.0, 0.5, 0.0],  # START → *
        [0,   0.1, 0.7, 0.0, 0.2],  # NOUN → *
        [0,   0.0, 0.1, 0.6, 0.3],  # VERB → *
        [0,   0.6, 0.3, 0.0, 0.1],  # DET → *
        [0,   0.0, 0.0, 0.0, 1.0],  # END  → END (absorbing)
    ])

    # Emission matrix T x V
    # rows = tag, cols = word
    emission = np.array([
        [0,   0,   0,   0   ],  # START emits nothing
        [0.1, 0.6, 0.0, 0.3 ],  # NOUN
        [0.0, 0.0, 0.9, 0.1 ],  # VERB
        [0.7, 0.0, 0.0, 0.3 ],  # DET
        [0,   0,   0,   0   ],  # END emits nothing
    ])

    return tags, vocab, transition, emission, T, V


def generate_dummy_test_sentences():
    # Three tiny tagged sentences
    # Format: list of (token, tag)

    s1 = [("the", "DET"), ("dog", "NOUN"), ("barks", "VERB")]
    s2 = [("a", "DET"), ("dog", "NOUN")]
    s3 = [("the", "DET"), ("dog", "NOUN"), ("barks", "VERB"), ("a", "DET"), ("dog", "NOUN")]

    return [s1, s2, s3]