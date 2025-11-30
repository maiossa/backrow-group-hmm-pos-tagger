
'''
HMM Speed Benchmark!
'''

# this allows me to import hmm module
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from hmm import HMMTagger
from hmm.dataio import load_treebank

print("start program")
start = time.perf_counter()
train = load_treebank("data/english/gum/dev.clean.conllu")
test  = load_treebank("data/english/gum/test.clean.conllu")
print(f"Loading time: {time.perf_counter() - start:.4f}s")
print("loaded treebanks")

tagger = HMMTagger()

# Training
print("start training")
start = time.perf_counter()
tagger.train(train)
print(f"Training time: {time.perf_counter() - start:.4f}s")
print("finish training")

# Evaluation
print("start evaluating")
start = time.perf_counter()
acc, micro, macro = tagger.evaluate(test)
print(f"Evaluation time: {time.perf_counter() - start:.4f}s")


# Single sentence tagging
sentence = test[0]
words = [tok["form"] for tok in sentence]

start = time.perf_counter()
tagger.tag(words)
print(f"Tag time (one sentence): {time.perf_counter() - start:.6f}s")
