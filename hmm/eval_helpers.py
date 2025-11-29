from collections import defaultdict


def accumulate_counts(true_tags, pred_tags, label_counts, tp_counts, fp_counts, fn_counts):
    """
        Update TP, FP, FN counts for each POS tag.
    """

    for t, p in zip(true_tags, pred_tags):
        label_counts[t] += 1

        if t == p:
            tp_counts[t] += 1
        else:
            fp_counts[p] += 1
            fn_counts[t] += 1


def compute_scores(label_counts, tp_counts, fp_counts, fn_counts):
    """
        Compute MICRO and MACRO F1 scores.
        MICRO = accuracy for single-label POS tagging.
        MACRO = average F1 over all tags (treats rare tags equally).
    """
    
    # ----- MICRO F1 -----
    micro_tp = sum(tp_counts.values())
    micro_fp = sum(fp_counts.values())
    micro_fn = sum(fn_counts.values())

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall    = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0

    if micro_precision + micro_recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    # ----- MACRO F1 -----
    f1_scores = []

    for label in label_counts:
        tp = tp_counts[label]
        fp = fp_counts[label]
        fn = fn_counts[label]

        # Skip tags with undefined precision or recall
        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)

        if precision + recall == 0:
            continue

        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return micro_f1, macro_f1


def update_accuracy(gold, pred, total_tokens, correct_tokens):
    for g, p in zip(gold, pred):
        total_tokens += 1
        if g == p:
            correct_tokens += 1
    return total_tokens, correct_tokens


def evaluate_hmm(hmm, test_sentences):
    label_counts = defaultdict(int)
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)

    total_tokens = 0
    correct_tokens = 0

    for sentence in test_sentences:
        words = [tok["form"] for tok in sentence]
        gold  = [tok["upos"] for tok in sentence]

        pred = hmm.tag(words)

        # If Viterbi returns (word, tag) tuples → use only tags
        if pred and isinstance(pred[0], tuple):
            pred = [p[1] for p in pred]

        # accuracy
        total_tokens, correct_tokens = update_accuracy(
            gold, pred, total_tokens, correct_tokens
        )

        # confusion counts
        accumulate_counts(
            gold, pred,
            label_counts, tp_counts, fp_counts, fn_counts
        )

        
    # final accuracy
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0

    # F1 scores
    micro_f1, macro_f1 = compute_scores(
        label_counts, tp_counts, fp_counts, fn_counts
    )

    return accuracy, micro_f1, macro_f1
