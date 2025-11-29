import re
from conllu import TokenList
from .tokens import START_TOKEN, END_TOKEN, UNK_TOKEN


def normalize_input(sentence, word2idx):
    # If the input is a string
    if isinstance(sentence, str):
        sentence = re.findall(r'\w+|[^\w\s]', sentence)

    # If the input is in UD's format (TokenList)
    if isinstance(sentence, TokenList):
        tokens = []
        for token in sentence:
            tokens.append(token["form"])
        sentence = tokens

    # Replace unseen words with UNK
    sentence = [tok if tok in word2idx else UNK_TOKEN for tok in sentence]

    # Add START + END tokens
    sentence = [START_TOKEN] + sentence + [END_TOKEN]

    return sentence


def pair_words_with_tags(sentence, predicted_tags):
    result = []
    for i in range(1, len(sentence) - 1):  # skip START/END
        original_word = sentence[i]
        predicted_tag = predicted_tags[i]
        result.append((original_word, predicted_tag))
    return result
