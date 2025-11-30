from collections import Counter, defaultdict
import pandas as pd
import random

from .tokens import START_TOKEN, END_TOKEN, START_TAG, END_TAG, UNK_TOKEN


def extract_tokens_tags(sentences, mask_rate = 0.0):
    # Flatten sentences into lists of tokens and tags
    tags = []
    tokens = []

    for sentence in sentences:
        tags.append(START_TAG)
        tokens.append(START_TOKEN)

        for token in sentence:
            tags.append(token["upos"])

            if mask_rate > 0 and random.random() < mask_rate:
                tokens.append(UNK_TOKEN)
            else:
                tokens.append(token["form"])

        tags.append(END_TAG)
        tokens.append(END_TOKEN)

    return tags, tokens


def replace_rare_tokens(tokens, threshold=1):
    # Replace less frequent words with UNK tokens
    token_counts = Counter(tokens)
    new_tokens = tokens[:]
    for i, token in enumerate(tokens):
        if token_counts[token] <= threshold:
            new_tokens[i] = UNK_TOKEN
    return new_tokens


def build_transition(tags):
    # Build transition matrix
    transition_counts = defaultdict(list)
    for i in range(len(tags) - 1):
        current_tag = tags[i]
        next_tag = tags[i + 1]
        transition_counts[current_tag].append(next_tag)

    # Normalize counts to create probability distribution
    transition_data = {}
    for tag, next_tags in transition_counts.items():
        tag_counter = Counter(next_tags)
        total = len(next_tags)
        prob_dist = {k: v / total for k, v in tag_counter.items()}
        transition_data[tag] = prob_dist

    # Ensure END_TAG doesn't transition anywhere
    transition_data[END_TAG] = {}

    # Create transition DataFrame
    df = pd.DataFrame(transition_data).T
    return df.fillna(0)


def build_emission(tags, tokens):
    # Build emission matrix
    emission_counts = defaultdict(list)
    for i in range(len(tokens)):
        current_tag = tags[i]
        current_token = tokens[i]
        emission_counts[current_token].append(current_tag)
    
     # Normalize counts to create probability distribution
    emission_data = {}
    for token, tag_list in emission_counts.items():
        tag_counter = Counter(tag_list)
        total = len(tag_list)
        prob_dist = {k: v / total for k, v in tag_counter.items()}
        emission_data[token] = prob_dist

    df = pd.DataFrame(emission_data)
    return df.fillna(0)


def reorder(transition_df, emission_df):
    # Create consistent tag ordering
    tag_order = sorted(transition_df.index)
    tag_order.remove(START_TAG)
    tag_order.remove(END_TAG)
    tag_order = tag_order + [START_TAG, END_TAG]

    transition_df = transition_df.reindex(index=tag_order, columns=tag_order, fill_value=0)
    emission_df   = emission_df.reindex(index=tag_order, fill_value=0)

    return transition_df, emission_df, tag_order
