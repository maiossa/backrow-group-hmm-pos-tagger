#!/usr/bin/env bash

# This script downloads the data from UD

BASE_DIR=data

REPOS=(
  UD_English-GUM
  UD_Spanish-AnCora
#   UD_Czech-CAC
#   UD_Slovak-SNK
#   UD_English-EWT
#   UD_Persian-PerDT
#   UD_Arabic-NYUAD
#   UD_German-GSD
#   UD_Urdu-UDTB
)

for repo in "${REPOS[@]}"; do
    language=$(echo "$repo" | cut -d'_' -f2 | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')
    treebank=$(echo "$repo" | cut -d'-' -f2 | tr '[:upper:]' '[:lower:]')
    
    echo "fetching $language ($treebank) ..."

    DEST=$BASE_DIR/$language/$treebank
    mkdir -p "$DEST"

    # list conllu files (train, test, dev)
    FILES=$(curl -s "https://api.github.com/repos/UniversalDependencies/$repo/contents" \
        | grep '"name": ".*conllu"' \
        | cut -d'"' -f4)

    # download each file directly
    for f in $FILES; do
        URL="https://raw.githubusercontent.com/UniversalDependencies/$repo/master/$f"
        filename="$DEST/$(echo "$f" | sed -E 's/.*-ud-//')"
        curl -L "$URL" -o "$filename"
        echo "  -> $f -> $filename"
    done
done
