from collections import defaultdict, Counter
from hmm.tokens import START_TOKEN, END_TOKEN, START_TAG, END_TAG, UNK_TOKEN, UNK_ID


class CorpusEncoder:

    def __init__(self, sentences=[], min_freq=0):
        '''
        min_freq: threshold to replace unfrequent words with UNK
        '''

        # indexings
        self.word2id = defaultdict(lambda: UNK_ID) | {UNK_TOKEN: UNK_ID, START_TOKEN: 1, END_TOKEN: 2}
        self.tag2id = {}

        self.vocab = set(self.word2id.keys())
        self.tagset = set()

        # id -> vocab and tagset
        self.words = list(self.vocab)
        self.tags = []

        self.min_freq = min_freq
        # removed words due to low frequency
        self.removed = Counter()

        if sentences:
            self.fit_transform(sentences)


    def _build_corpus(self, sentences):

        # add start and end tokens
        sentences = [[(START_TOKEN, START_TAG)] + sen + [(END_TOKEN, END_TAG)] for sen in sentences]

        # flatten all sentences into a list, and seperate words and tags
        flat = [pair for sen in sentences for pair in sen]
        words, labels = zip(*flat)

        words = self._replace_rare_words(words)

        vocab = set(words)
        tagset = set(labels)
        
        return words, labels, vocab, tagset


    def _replace_rare_words(self, words):
        # replace less frequent words with UNK tokens
        
        threshold = self.min_freq
        removed = self.removed
        
        counts = Counter(words) + removed
        rare = {t for t,c in counts.items() if c <= threshold}
        removed.update(rare)

        return [UNK_TOKEN if t in rare else t for t in words]


    def fit_transform(self, sentences):
        '''
        sentences: list of (word, tag)
        '''

        # inserts new vocab and tags alongside old data, and returns the encoded version
        tokens, labels, vocab, tags = self._build_corpus(sentences)
        self._extend_index(vocab, tags)
        return self.encode_data(tokens, labels)


    def _extend_index(self, new_vocab: set, new_tags: set):
        diff_vocab = new_vocab - self.vocab
        diff_tags = new_tags - self.tagset

        # update to have the new word and tag indexes added
        self.word2id |= {word: idx for idx, word in enumerate(diff_vocab, start=len(self.vocab))}
        self.tag2id |= {tag: idx for idx, tag in enumerate(diff_tags, start=len(self.tags))}

        self.vocab.update(new_vocab)
        self.tagset.update(new_tags)

        self.words.extend(diff_vocab)
        self.tags.extend(diff_tags)


    def encode_words(self, tokens):
        w2i = self.word2id
        return [w2i[tok] for tok in tokens]


    def encode_tags(self, tags):
        t2i = self.tag2id
        return [t2i[tag] for tag in tags]


    def encode_data(self, tokens, tags):
        return self.encode_words(tokens), self.encode_tags(tags)


    def decode_tokens(self, token_ids):
        i2w = self.idx2word
        return [i2w[i] for i in token_ids]

    def decode_tags(self, tag_ids):
        i2t = self.idx2tag
        return [i2t[i] for i in tag_ids]


if __name__ == '__main__':
        
    from hmm_fast.dataio import load_treebank

    sentences = load_treebank("data/english/gum/dev.conllu")

    enc = CorpusEncoder(min_freq=0)

    enc.fit_transform(sentences[0:20])
    # print(enc.words)

    print(
        enc.fit_transform(sentences[0:1])
    )
    # print(enc.words)


    