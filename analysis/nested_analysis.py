import json
import numpy as np
import argparse
import os

def expand_discontinuous(entity):
    text, offsets = entity
    expanded = list(zip(text, offsets))
    return [[[text], [offset]] for text, offset in expanded]

def is_discontinuous(entity):
    text, offsets = entity
    if len(offsets) > 1:
        return True
    return False

def is_nested(offsets1, offsets2):
    # Checks if offsets1 is nested in offsets2
    # NOTE!! It does not check the other way around
    # Currently only works with continuous spans
    # TODO: extend to discontinous spans
    if len(offsets1) > 0 and len(offsets2) > 0:
        if len(offsets1) == 1 and len(offsets2) == 1:
            bidx1, eidx1 = np.array(offsets1[0].split(":"), dtype=int)
            bidx2, eidx2 = np.array(offsets2[0].split(":"), dtype=int)
        # check if one of the spans is contained within the other
        if bidx1 >= bidx2 and eidx1 < eidx2:
            return True
        elif bidx1 > bidx2 and eidx1 <= eidx2:
            return True
        else:
            return False
    else:
        return False

def get_nested(sent):
    nested = []
    opinions = sent["opinions"]
    labels = {"Source": [],
              "Target": [],
              "Polar_expression": []
              }
    if len(opinions) > 0:
        # Get all the labels of type 'label'
        for opinion in opinions:
            for label in ["Source", "Target", "Polar_expression"]:
                opinion_entity = opinion[label]
                if is_discontinuous(opinion_entity):
                    expanded_entities = expand_discontinuous(opinion_entity)
                    for oe in expanded_entities:
                        if oe not in labels[label]:
                            labels[label].append(oe)
                else:
                    if opinion_entity not in labels[label]:
                        labels[label].append(opinion_entity)
        # check if they are nested
        for label1 in ["Source", "Target", "Polar_expression"]:
            for label2 in ["Source", "Target", "Polar_expression"]:
                ents1 = labels[label1]
                ents2 = labels[label2]
                for ent1 in ents1:
                    for ent2 in ents2:
                        off1 = ent1[1]
                        off2 = ent2[1]
                        try:
                            if is_nested(off1, off2):
                                nested.append((label1, label2, ent1, ent2))
                        except UnboundLocalError:
                            print(ent1)
                            print(ent2)
                            print("-------")
    return labels, nested



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/norec_fine")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "train.json")) as o:
        train = json.load(o)

    with open(os.path.join(args.data_dir, "dev.json")) as o:
        dev = json.load(o)

    with open(os.path.join(args.data_dir, "test.json")) as o:
        test = json.load(o)

    overall_count = {"Source": 0,
                     "Target": 0,
                     "Polar_expression": 0
                     }

    nested_count = {"Source": {"Source": 0,
                         "Target": 0,
                         "Polar_expression": 0
                         },
              "Target": {"Source": 0,
                         "Target": 0,
                         "Polar_expression": 0
                         },
              "Polar_expression": {"Source": 0,
                                   "Target": 0,
                                   "Polar_expression": 0
                                   }
              }

    for sent in train + dev + test:
        entities, nested = get_nested(sent)
        for label, ents in entities.items():
            for ent in ents:
                if ent != [[[], []]]:
                    overall_count[label] += 1

        if len(nested) > 0:
            for label1, label2, ent1, ent2 in nested:
                nested_count[label1][label2] += 1


    print("Nested analysis of {}".format(args.data_dir))
    print("#" * 40)
    for label in overall_count.keys():
        ncount = sum(nested_count[label].values())
        npercent = ncount / overall_count[label] * 100

        print("Number nested {0}: {1}".format(label, ncount))
        print("Percent nested {0}:  {1:.1f}%".format(label, npercent))
        print()
