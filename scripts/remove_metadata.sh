#!/usr/bin/env bash
set -e

BASE_DIR="data"

echo "Cleaning UD treebanks: removing metadata (#...) and empty lines"
echo

# Loop through all .conllu files
find "$BASE_DIR" -type f -name "*.conllu" | while read -r file; do
    cleaned="${file%.conllu}.clean.conllu"

    echo " → Cleaning $file"
    echo "   saving to $cleaned"

    # Remove metadata and blank lines
    grep -v '^#' "$file" | sed '/^\s*$/d' > "$cleaned"
done

echo
echo "Done. Cleaned files saved as *.clean.conllu"
