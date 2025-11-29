# Before experimenting with the notebook, I just want to make sure everything works as intended

from hmm_tagger import HMMTagger
from process_data import get_data

########################################################
# English
########################################################

train_sentences = get_data("data/english/gum/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/english/gum/test.conllu")
print("#########################")
print("ENGLISH TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")

########################################################
# Czech
########################################################

train_sentences = get_data("data/czech/cac/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/czech/cac/test.conllu")
print("#########################")
print("CZECH TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")

########################################################
# Arabic
########################################################

train_sentences = get_data("data/arabic/nyuad/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/arabic/nyuad/test.conllu")
print("#########################")
print("ARABIC TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")

########################################################
# German 
########################################################

train_sentences = get_data("data/german/gsd/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/german/gsd/test.conllu")
print("#########################")
print("GERMAN TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")


########################################################
# Persian
########################################################

train_sentences = get_data("data/persian/perdt/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/persian/perdt/test.conllu")
print("#########################")
print("PERSIAN TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")


########################################################
# Slovak 
########################################################

train_sentences = get_data("data/slovak/snk/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/slovak/snk/test.conllu")
print("#########################")
print("SLOVAK TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")


########################################################
# Spanish 
########################################################

train_sentences = get_data("data/spanish/ancora/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/spanish/ancora/test.conllu")
print("#########################")
print("SPANISH TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")


########################################################
# Urdu 
########################################################

train_sentences = get_data("data/urdu/udtb/train.conllu")

tagger = HMMTagger()
tagger.train(train_sentences)

test_sentences =  get_data("data/urdu/udtb/test.conllu")
print("#########################")
print("URDU TEST:")
print("#########################")
print(tagger.evaluate(test_sentences))
print("")




