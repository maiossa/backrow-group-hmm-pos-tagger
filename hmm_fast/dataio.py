
def fast_parse_conllu(path):
    """
    Manual CoNLL-U parser.
    Returns: a lazy list of sentences, each sentence = list of dicts:
             {"id": int, "form": str, "upos": str}
    
    I used conllu library's source as a reference before writing this
    """
    sentences = []
    current = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            # empty line -> next sentence
            if len(line) == 1:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # skip all metadata
            if line[0] == "#":
                continue

            parts = line.split("\t", 4)
            if len(parts) < 4:
                # malformed
                continue

            token_id = parts[0]

            # skip multiword tokens like 3-5 or 3.1
            # for example, in train data, we have participant's beliefs, and we have two lines for participants
            if "-" in token_id or "." in token_id:
                continue

            token_id = int(token_id)

            form = parts[1]
            upos = parts[3]

            # current.append({"id": token_id, "form": form, "upos": upos})
            current.append((form, upos))

    # if file does not end with newline
    if current:
        sentences.append(current)

    return sentences



def load_treebank(path):
    """
    load a CoNLL-U file and return its sentences.

    args:
        path (str): path to the .conllu file.

    returns:
        tuple(str, str): parsed sentences from the Universal Dependencies dataset.
    """

    return fast_parse_conllu(path)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    data = load_treebank("data/english/gum/train.conllu")
    print(f"Loading time: {time.perf_counter() - start:.4f}s")
    print(data[0])