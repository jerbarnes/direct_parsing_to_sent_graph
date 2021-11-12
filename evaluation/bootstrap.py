import argparse
import json
import time
from numba.core.decorators import njit
import numpy as np
from typing import Tuple, List, TypeVar, Optional
from evaluate_single_dataset import evaluate
from evaluate import (convert_opinion_to_tuple, get_flat, sent_tuples_in_list,
                      weighted_score)

A = TypeVar("A")


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def span_counts(gold_sent: List[A],
                pred_sent: List[A],
                test_label: str = "Source") -> Tuple[int, int, int]:
    """Takes in one sentence-pair and counts tp, fp, fn"""
    tp, fp, fn = 0, 0, 0
    gold_labels = get_flat(gold_sent, test_label)
    pred_labels = get_flat(pred_sent, test_label)
    for gold_label, pred_label in zip(gold_labels, pred_labels):
        # TP
        if gold_label == pred_label == test_label:
            tp += 1
        # FP
        if gold_label != test_label and pred_label == test_label:
            fp += 1
        # FN
        if gold_label == test_label and pred_label != test_label:
            fn += 1
    return tp, fp, fn


def tuple_counts(gtuples,
                 ptuples,
                 keep_polarity=True,
                 weighted=True) -> Tuple[float, float, int, int, int, int]:
    """
    Takes in one sentence-pair's tuples and counts weighted-tp, tp, fp, fn
    """
    weighted_tp_p = 0
    tp_r = 0
    weighted_tp_r = 0
    tp_p = 0
    fp = 0
    fn = 0
    for stuple in gtuples:
        if sent_tuples_in_list(stuple, ptuples, keep_polarity):
            if weighted:
                weighted_tp_r += weighted_score(stuple, ptuples)
                tp_r += 1
            else:
                weighted_tp_r += 1
                tp_r += 1
        else:
            fn += 1
    for stuple in ptuples:
        if sent_tuples_in_list(stuple, gtuples, keep_polarity):
            if weighted:
                weighted_tp_p += weighted_score(stuple, gtuples)
                tp_p += 1
            else:
                weighted_tp_p += 1
                tp_p += 1
        else:
            fp += 1
    # print(weighted_tp, weighted_tp_, tp, tp_, fp, fn)
    # assert tp == tp_ and weighted_tp == weighted_tp_
    return weighted_tp_r, weighted_tp_p, tp_r, tp_p, fp, fn


def read_data(gold_fn: str, pred_fn: str) -> Tuple[List[float], int]:
    """
    reads the gold and pred files, computes counts for different measures,
    and returns the number of sentences, and the list with counts
    """
    with open(gold_fn) as f:
        gold = json.load(f)

    with open(pred_fn) as f:
        pred = json.load(f)

    tgold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    tpred = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in pred])

    L: List[float] = []

    # for every sentence
    # for every measure
    # get counts and put them into one long row
    for s_i in range(len((gold))):
        # print(s_i, gold[s_i], s_id)
        assert gold[s_i]["sent_id"] == pred[s_i]["sent_id"]
        for label in ["Source", "Target", "Polar_expression"]:
            tp, fp, fn = span_counts(gold[s_i], pred[s_i], test_label=label)
            L.extend([tp, fp, fn])
        s_id = gold[s_i]["sent_id"]
        wtp_r, wtp_p, tp_r, tp_p, fp, fn = tuple_counts(tgold[s_id],
                                                        tpred[s_id],
                                                        keep_polarity=False)
        L.extend([wtp_r, wtp_p, tp_r, tp_p, fp, fn])
        wtp_r, wtp_p, tp_r, tp_p, fp, fn = tuple_counts(
            tgold[s_id], tpred[s_id])
        L.extend([wtp_r, wtp_p, tp_r, tp_p, fp, fn])
    return L, len(gold)


def prec(x: np.array, i: int, j: int, k: Optional[int] = None) -> np.array:
    # when calculating weighted precision we need an extra variable
    if k is None:
        k = i
    return np.divide(x[..., k], (x[..., i] + x[..., j] + 1e-6))


def rec(x: np.array, i: int, j: int, k: Optional[int] = None) -> np.array:
    # when calculating weighted precision we need an extra variable
    if k is None:
        k = i
    return np.divide(x[..., k], (x[..., i] + x[..., j] + 1e-6))


def fscore(p: np.array, r: np.array) -> np.array:
    return np.divide(2 * p * r, (p + r + 1e-6))


def compute_scores(scores: np.array, p: np.array, r: np.array,
                   x: int) -> np.array:
    scores[:, x] = fscore(p, r)
    return scores


def fill_scores(scores, evals, debug=False):
    eval_cnts = range(evals.shape[-1])

    # terrible hardcoded for the 5 measures using 3/4 values to calculate F1
    # those are my measures
    # tp, fp, fn
    # _, _, source_f1 = span_f1(gold, preds, test_label="Source")
    # _, _, target_f1 = span_f1(gold, preds, test_label="Target")
    # _, _, expression_f1 = span_f1(gold, preds, test_label="Polar_expression")

    # weighted_tp, tp, fp, fn
    # _, _, unlabeled_f1 = tuple_f1(tgold, tpreds, keep_polarity=False)
    # _, _, f1 = tuple_f1(tgold, tpreds)

    l_i = 0
    for f_i in range(5):  # there are 5 eval measures
        if f_i < 3:
            tp, fp, fn = eval_cnts[l_i:l_i + 3]
            p = prec(evals, tp, fp)
            r = rec(evals, tp, fn)
            compute_scores(scores, p, r, f_i)
            l_i += 3
        elif f_i >= 3:
            wtp_r, wtp_p, tp_r, tp_p, fp, fn = eval_cnts[l_i:l_i + 6]
            p = prec(evals, tp_p, fp, wtp_p)
            r = rec(evals, tp_r, fn, wtp_r)
            if debug:
                print("prec", evals[..., wtp_p], evals[..., tp_p], evals[...,
                                                                         fp])
                print("rec", evals[..., wtp_r], evals[..., tp_r], evals[...,
                                                                        fn])
            compute_scores(scores, p, r, f_i)
            l_i += 6
    return scores


def bootstrap(gold: str,
              pred_a: str,
              pred_b: str,
              b: int = 1,
              debug: bool = False
              ) -> None:  # Dict[str, Tuple[float, float, float]]:
    if debug:
        s = time.time()

    n_measures = 5  # number of measures
    L1, n = read_data(gold, pred_a)
    L2, _ = read_data(gold, pred_b)
    assert n == _

    # number of runs
    b = int(b)

    n_features = int(len(L1) / n)

    if debug:
        print(f"reading in data {time.time() - s}")
        s = time.time()

    M1 = np.array(L1).reshape(int(len(L1) / n_features), n_features)
    M2 = np.array(L2).reshape(int(len(L2) / n_features), n_features)

    # sample 'b' ids for a test set of size 'n' with 'r' runs
    # np.random.choice(n * r, n*b).reshape(b, n)

    if debug:
        print(f"data as matrix {time.time() - s}")
        s = time.time()

    # sample_ids samples b datasets of size n with indices ranging the five
    # runs creating a sample out of all runs
    @njit
    def get_sample_ids(n, b):
        return np.random.choice(n, n * b).reshape(b, n)

    sample_ids = get_sample_ids(n, b)

    if debug:
        print(f"get samples {time.time() - s}")
        s = time.time()

    # fill a zero matrix with how often each sentence was drawn in one sample
    # with b=1e5 numba's jit reduces the runtime for this step
    # from 80 to 1-2 seconds
    @njit
    def get_samples(n, b, sample_ids):
        samples = np.zeros((n, b))
        for j in range(b):
            for i in range(n):
                samples[sample_ids[j, i], j] += 1
        return samples

    samples = get_samples(n, b, sample_ids)

    if debug:
        print(f"samples to right format {time.time() - s}")
        s = time.time()

    # get the counts for the sample
    # Mx has the counts per sentence and samples chooses how often each sample
    # is taken resulting in a matrix of b rows with sums of tp, fp, fn etc.
    evals1 = (np.einsum('ik,il->lk', M1, samples))
    evals2 = (np.einsum('ik,il->lk', M2, samples))

    if debug:
        print(f"extract sample counts {time.time() - s}")
        s = time.time()

    # compute the eval measures for each row/sample
    sample_scores1 = fill_scores(np.zeros((b, n_measures)), evals1, False)
    sample_scores2 = fill_scores(np.zeros((b, n_measures)), evals2, False)

    if debug:
        print(f"compute scores {time.time() - s}")
        s = time.time()

    # scores for the dataset across all runs
    true_scores1 = fill_scores(np.zeros((1, n_measures)), np.sum(M1, axis=0))
    true_scores2 = fill_scores(np.zeros((1, n_measures)), np.sum(M2, axis=0))

    # bootstrap scores
    deltas = true_scores1 - true_scores2
    deltas *= 2

    diffs = sample_scores1 - sample_scores2
    diffs_plus = np.where(diffs >= 0, diffs, 0)
    diffs_minus = np.where(diffs < 0, diffs, 0)

    deltas_plus = np.where(deltas > 0, deltas, np.float("inf"))

    deltas_minus = np.where(deltas < 0, deltas, -np.float("inf"))
    s1 = np.sum(diffs_plus > deltas_plus, axis=0)
    s2 = np.sum(diffs_minus < deltas_minus, axis=0)

    if debug:
        print(f"the rest {time.time() - s}")

    if debug:
        print(true_scores1)
        print(true_scores2)

        print(s1 / b)
        print(s2 / b)

        print()
    s1 = s1 / b
    s2 = s2 / b
    end = color.END

    print(
        f"{color.BOLD}{color.BLUE}{pred_a} || {color.RED}{pred_b}{color.END}")

    measures = [
        "source/f1", "target/f1", "expression/f1",
        "sentiment_tuple/unlabeled_f1", "sentiment_tuple/f1"
    ]
    for i, name in enumerate(measures):
        x = true_scores1[0][i]
        y = true_scores2[0][i]
        z = s1[i] if x > y else s2[i]
        if z < 0.05 and x > y:
            bold = color.BLUE
        elif z < 0.05 and y > x:
            bold = color.RED
        else:
            bold = color.END
        print(f"{bold}{name:<13}: {x:.2%}\t{y:.2%}\t{z:.4f}{end}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", help="gold json file")
    parser.add_argument("--pred_file_a", help="prediction json file a)")
    parser.add_argument("--pred_file_b", help="prediction json file b)")
    parser.add_argument("--b",
                        help="number of resampling 'b' for bootstrap",
                        type=float)
    parser.add_argument("--debug", help="debug prints", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    results = evaluate(args.gold_file, args.pred_file_a)
    print(json.dumps(results, indent=2))
    print()
    print(list(results.values()))

    results = evaluate(args.gold_file, args.pred_file_b)
    print(json.dumps(results, indent=2))
    print()
    print(list(results.values()))

    bootstrap(args.gold_file, args.pred_file_a, args.pred_file_b, args.b,
              args.debug)


if __name__ == "__main__":
    main()
