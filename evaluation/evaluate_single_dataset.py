import json
from evaluate import convert_opinion_to_tuple, tuple_f1, span_f1
import argparse


def evaluate(gold_file, pred_file):
    with open(gold_file) as o:
        gold = json.load(o)

    with open(pred_file) as o:
        preds = json.load(o)

    tgold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    tpreds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    g = sorted(tgold.keys())
    p = sorted(tpreds.keys())

    if g != p:
        print("Missing some sentences!")
        return 0.0, 0.0, 0.0

    _, _, source_f1 = span_f1(gold, preds, test_label="Source")
    _, _, target_f1 = span_f1(gold, preds, test_label="Target")
    _, _, expression_f1 = span_f1(gold, preds, test_label="Polar_expression")

    _, _, unlabeled_f1 = tuple_f1(tgold, tpreds, keep_polarity=False)
    prec, rec, f1 = tuple_f1(tgold, tpreds)

    print("Source F1: {0:.3f}".format(source_f1))
    print("Target F1: {0:.3f}".format(target_f1))
    print("Expression F1: {0:.3f}".format(expression_f1))
    print("Unlabeled Sentiment Tuple F1: {0:.3f}".format(unlabeled_f1))
    print("Sentiment Tuple F1: {0:.3f}".format(f1))
    return prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")

    args = parser.parse_args()

    _, _, f1 = evaluate(args.gold_file, args.pred_file)


if __name__ == "__main__":
    main()
