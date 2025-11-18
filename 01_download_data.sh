#!/usr/bin/env bash

# This script downloads the data from UD

mkdir -p "./data/czech"

curl -L \
  "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/master/cs_cac-ud-train.conllu" \
  -o "./data/czech/cs_cac-ud-train.conllu"


# Czech
# https://github.com/UniversalDependencies/UD_Czech-CAC

# Slovak
# https://github.com/UniversalDependencies/UD_Slovak-SNK/tree/master

# English
# https://github.com/UniversalDependencies/UD_English-GUM/tree/master

# Spanish
# https://github.com/UniversalDependencies/UD_Spanish-AnCora/tree/master

# Persian
# https://github.com/UniversalDependencies/UD_Persian-PerDT/tree/master

# Arabic
# https://github.com/UniversalDependencies/UD_Arabic-NYUAD/tree/master

# German
# https://github.com/UniversalDependencies/UD_German-GSD/tree/master

# Urdu
# https://github.com/UniversalDependencies/UD_Urdu-UDTB/tree/master


