from process_data import get_data
from hmm_tagger import HMMTagger

if __name__ != '__main__':
    quit(0)

sentences = get_data("data/english/gum/dev.conllu")
tagger = HMMTagger()
tagger.train(sentences)
tagger.save_model('model')

tagger = tagger.load_model('model')
print(tagger.transition_matrix)