from process_data import get_data
from hmm_tagger import HMMTagger
from evaluation import evaluate
if __name__ != '__main__':
    quit(0)

sentences = get_data("data/english/gum/dev.conllu")
tagger = HMMTagger()
tagger.train(sentences)
tagger.save_model('model')

tagger = tagger.load_model('model')

print(tagger.transition_matrix)
test_data = get_data("data/english/gum/test.conllu")

acc, micro, macro = evaluate(tagger, test_data)
print("Accuracy:", acc)
print("Micro F1:", micro)
print("Macro F1:", macro)