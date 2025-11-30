from hmm.dataio import load_treebank
from hmm.model import HMMTagger

if __name__ != '__main__':
    quit(0)


train_sentences = load_treebank("data/english/gum/train.conllu")
tagger = HMMTagger()
tagger.train(train_sentences, mask_rate =0.01)

test_sentences =  load_treebank("data/english/gum/test.conllu")
print(tagger.evaluate(test_sentences))
