from conllu import parse

def get_data(path):

    with open(path, "r", encoding="utf8") as f:
        data = f.read()

    sentences = parse(data)

    return sentences

