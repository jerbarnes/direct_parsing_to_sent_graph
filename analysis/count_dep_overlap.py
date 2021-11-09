from collections import Counter
from pprint import pprint
from typing import Iterable, List, Dict, Tuple


def read_sentences(fn: str) -> Iterable[Tuple[str, List[List[str]]]]:
    """reads conllu file and yields sentences: conllu rows"""
    with open(fn) as fh:
        sentence: List[List[str]] = []
        for line in fh:
            if line.startswith("# sent_id"):
                sid = line.split("=")[1].strip()
            elif line.startswith("# text"):
                continue
            elif line.strip() == "":
                yield sid, sentence
                sentence = []
            else:
                sentence.append(line.split())


def get_arcs(sentence: List[List[str]]) -> Dict[int, List[str]]:
    """reads conllu-sentence and returns 'node -> list of arcs' -mapping"""
    arcs: Dict[int, List[str]] = {}
    for token in sentence:
        arcs[int(token[0])] = []
        if token[-1] != "_":
            arcs[int(token[0])].extend(token[-1].split("|"))
    return arcs


def overlapping(arcs: Dict[int, List[str]]) -> List[Tuple[str, ...]]:
    """takes node-arcs mapping and returns list with overlapping tuples"""
    over = []
    for node in arcs:
        aa = [a.split(":") for a in arcs[node]]
        tmp = aa[:]
        for a in aa:
            # all arcs with the same source node overlap
            os = [x for x in tmp if x[0] == a[0]]
            # remove those from the pool
            for o in os:
                tmp.remove(o)
            # when there is an overlap: 2 or more
            if len(os) > 1:
                _, lo = zip(*os)
                over.append(lo)
    return over


def count_overlap(fn: str) -> Tuple[Dict[str, int],
                                    Dict[Tuple[str,...], int],
                                    Dict[Tuple[str,...], int]]:
    counter: Dict[Tuple[str, ...], int] = Counter()
    total: Dict[str, int] = Counter()
    for _, sentence in read_sentences(fn):
        arcs = get_arcs(sentence)
        # update total
        for n in arcs:
            for a in arcs[n]:
                total[a.split(":")[1]] += 1
        # end update total
        counter.update(overlapping(arcs))
    # sc is a simplified counter as there might be tuples of overlapping arcs
    # greater than two
    sc: Dict[Tuple[str, ...], int] = Counter()
    for k in counter:
        sc[tuple(set(k))] += counter[k]
    return total, counter, sc  # total, counter, and simple counter


def main():
    # I mark hardcoded bits that might differ between anyone's setup with WARNING
    # WARNING: different languages names maybe?
    langs = "ca ds_unis en es eu mpqa norec_fine".split()
    # change splits to only look at one split
    splits = "train dev test".split()
    for l in langs:
        cs = []
        scs = []
        totals = []
        for tdt in splits:
            # WARNING: I have for each 'lang' a folder with head_final with tdt.conllu
            fn = f"..//data/{l}/head_final/{tdt}.conllu"
            # total is the total number of arcs, c is overlapping instances
            # and sc is overlapping istances simplified (target, target, target) -> (target)
            total, c, sc = count_overlap(fn)
            cs.append(c)
            scs.append(sc)
            totals.append(total)
            # print for every split
            # print(tdt, sc)
        C = Counter()
        for c in cs:
            C.update(c)
        SC = Counter()
        for sc in scs:
            SC.update(sc)
        T = Counter()
        for t in totals:
            T.update(t)
        print(l)
        pprint(T)
        # pprint(SC)
        pprint(C)
        t = sum(T.values())
        # count the number of arcs that each overlap "costs"
        # print("Number of lost arcs")
        lost = 0
        for k, v in sorted(C.items(), key=lambda x: -x[1]):
            # when there are 10 instances of 3 arcs overlapping we lose 2*10 = 3*10 - 10 instances
            x = len(k) * v - v
            # print("\t", k, x)
            lost += x
        print(f"Number of lost arcs in total: {lost}")
        print(f"Number of arcs in total: {t}")
        print(f"Percentage lost {lost / t:.2%}")
        # another dictionary with probabilities instead of counts
        # pSC = Counter({k: round(v/t, 4) for k, v in SC.items()})
        # pprint(pSC)
        print()


if __name__ == "__main__":
    print(
        "If something does not work you might have to change the code as paths are hardcoded"
    )
    main()