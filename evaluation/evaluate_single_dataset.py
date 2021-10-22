import json
from evaluate import convert_opinion_to_tuple, tuple_f1
import argparse


def evaluate(gold_file, pred_file):
    with open(gold_file) as o:
        gold = json.load(o)

    with open(pred_file) as o:
        preds = json.load(o)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    g = sorted(gold.keys())
    p = sorted(preds.keys())

    if g != p:
        print("Missing some sentences!")
        return 0.0, 0.0, 0.0

    prec, rec, f1 = tuple_f1(gold, preds)
    return prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")

    args = parser.parse_args()

    _, _, f1 = evaluate(args.gold_file, args.pred_file)

    print("Sentiment Tuple F1: {0:.3f}".format(f1))


if __name__ == "__main__":
    main()
