import numpy as np


def save_hmm_params(filepath, transition_matrix, emission_matrix):
    
    # ensure correct file extension
    if not filepath.endswith(".npz"):
        filepath += ".npz"

    np.savez(filepath,
        transition=transition_matrix,
        emission=emission_matrix)


def load_hmm_params(filepath):
    
    # ensure correct file extension
    if not filepath.endswith(".npz"):
        filepath += ".npz"

    data = np.load(filepath, allow_pickle=True)
    transition = data["transition"]
    emission = data["emission"]
    
    return transition, emission

