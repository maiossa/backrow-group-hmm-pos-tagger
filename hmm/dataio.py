from conllu import parse


def load_treebank(path):
    """
    load a CoNLL-U file and return its sentences.

    args:
        path (str): path to the .conllu file.

    returns:
        list[TokenList]: parsed sentences from the Universal Dependencies dataset.
    """

    with open(path, "r", encoding="utf8") as f:
        data = f.read()

    sentences = parse(data)

    return sentences

