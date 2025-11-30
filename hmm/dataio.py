from conllu import parse_incr


def lazy_load_treebank(path):
    # conllu always expects id field
    fields= ["id", "form", "upos"]

    with open(path, "r", encoding="utf8") as f:
        # using generators and selected fields to make loading much faster 
        for sent in parse_incr(f, fields=fields):
            
            yield sent
            # yield [
            #     {key: token[key] for key in ['upos', 'form']} for token in sent
            # ]


def load_treebank(path):
    """
    load a CoNLL-U file and return its sentences.

    args:
        path (str): path to the .conllu file.

    returns:
        list[TokenList]: parsed sentences from the Universal Dependencies dataset.
    """

    return lazy_load_treebank(path)
    