from conllu import parse_incr, parse


def lazy_load_treebank(path):
    # conllu always expects id field
    fields= ["id", "form", "upos"]

    with open(path, "r", encoding="utf8") as f:
        for sentence in parse_incr(f, fields=fields):
            yield sentence


def load_treebank_in_ram(path):
    with open(path, "r", encoding="utf8") as f:
        data = f.read()
        return parse(data, fields=["id", "form", "upos"])


def load_treebank(path, parse_incrementally=False):
    """
    load a CoNLL-U file and return its sentences.

    args:
        path (str): path to the .conllu file.

    returns:
        list[TokenList]: parsed sentences from the Universal Dependencies dataset.
    """

    if parse_incrementally:
        return lazy_load_treebank(path)
    
    return load_treebank_in_ram(path)