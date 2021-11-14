import json
import argparse
from src.col_data import read_col_data

def find_roots(col_sent):
    roots = []
    for token in col_sent.tokens:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx == 0:
                    roots.append(token)
    return list(set(roots))

def sort_tokens(tokens):
    sorted_tokens = []
    sorted_idxs = sorted([token.id for token in tokens])
    for idx in sorted_idxs:
        for token in tokens:
            if token.id == idx:
                sorted_tokens.append(token)
    return sorted_tokens

def get_char_offsets(sorted_tokens):
    char_offsets = []
    idxs = []
    current_idxs = []
    current_bidx = None
    current_eidx = None
    for i, token in enumerate(sorted_tokens):
        bidx, eidx = token.char_offsets
        if current_bidx == None:
            current_bidx = bidx
            current_idxs.append(i)
        if current_eidx == None:
            current_eidx = eidx
            if i not in current_idxs:
                current_idxs.append(i)
        elif eidx > current_eidx and bidx == current_eidx + 1:
            current_eidx = eidx
            if i not in current_idxs:
                current_idxs.append(i)
        else:
            char_offsets.append((current_bidx, current_eidx))
            idxs.append(current_idxs)
            current_idxs = [i]
            current_bidx = bidx
            current_eidx = eidx
    char_offsets.append((current_bidx, current_eidx))
    idxs.append(current_idxs)
    return char_offsets, idxs


def gather_expressions(roots, col_sent):
    expression_tokens = []
    expressions = []
    # find all other expression tokens
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if "exp" in label and token not in roots:
                    expression_tokens.append(token)
    # group them by root
    for root in roots:
        exps = [root]
        for token in expression_tokens:
            for idx, label in token.scope:
                if idx == root.id and token not in exps:
                    exps.append(token)
        # sort them by token id
        exp = sort_tokens(exps)
        # get the char_offsets and token ids for each
        char_offset, token_groups = get_char_offsets(exp)
        # convert everything to strings following json sent_graph format
        tokens = []
        char_offsets = []
        for token_group in token_groups:
            token_string = ""
            for i in token_group:
                token_string += exp[i].form + " "
            tokens.append(token_string.strip())
        for bidx, eidx in char_offset:
            char_offsets.append("{0}:{1}".format(bidx, eidx))
        expressions.append([tokens, char_offsets])
    return expressions

def gather_targets(roots, col_sent):
    targets = []
    # find all target roots
    exp_root_idxs = dict([(token.id, {}) for token in roots])
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx in exp_root_idxs and "targ" in label:
                    exp_root_idxs[idx][token.id] = [token]
                    for token2 in col_sent:
                        if len(token2.scope) > 0:
                            for idx2, label2 in token2.scope:
                                if idx2 == token.id and token2 not in exp_root_idxs[idx][token.id]:
                                    exp_root_idxs[idx][token.id].append(token2)
    for root_idx, target_group in exp_root_idxs.items():
        root_targets = []
        for target_idx, target_tokens in target_group.items():
            target_tokens = sort_tokens(target_tokens)
            char_offset, token_groups = get_char_offsets(target_tokens)
            # convert everything to strings following json sent_graph format
            tokens = []
            char_offsets = []
            for token_group in token_groups:
                token_string = ""
                for i in token_group:
                    token_string += target_tokens[i].form + " "
                tokens.append(token_string.strip())
            for bidx, eidx in char_offset:
                char_offsets.append("{0}:{1}".format(bidx, eidx))
            root_targets.append([tokens, char_offsets])
        if len(root_targets) > 0:
            targets.append(root_targets)
        else:
            targets.append([[[], []]])
    return targets

def gather_holders(roots, col_sent):
    holders = []
    # find all target roots
    exp_root_idxs = dict([(token.id, {}) for token in roots])
    for token in col_sent:
        if len(token.scope) > 0:
            for idx, label in token.scope:
                if idx in exp_root_idxs and "holder" in label:
                    exp_root_idxs[idx][token.id] = [token]
                    for token2 in col_sent:
                        if len(token2.scope) > 0:
                            for idx2, label2 in token2.scope:
                                if idx2 == token.id and token2 not in exp_root_idxs[idx][token.id]:
                                    exp_root_idxs[idx][token.id].append(token2)
    for root_idx, holder_group in exp_root_idxs.items():
        root_holders = []
        for holder_idx, holder_tokens in holder_group.items():
            holder_tokens = sort_tokens(holder_tokens)
            char_offset, token_groups = get_char_offsets(holder_tokens)
            # convert everything to strings following json sent_graph format
            tokens = []
            char_offsets = []
            for token_group in token_groups:
                token_string = ""
                for i in token_group:
                    token_string += holder_tokens[i].form + " "
                tokens.append(token_string.strip())
            for bidx, eidx in char_offset:
                char_offsets.append("{0}:{1}".format(bidx, eidx))
            root_holders.append([tokens, char_offsets])
        if len(root_holders) > 0:
            holders.append(root_holders)
        else:
            holders.append([[[], []]])
    return holders

def get_polarities(roots):
    polarities = []
    for root in roots:
        polarity = None
        for idx, label in root.scope:
            if "exp" in label:
                polarity = label.split("-")[1]
        polarities.append(polarity)
    return polarities

def convert_col_sent_to_json(col_sent):
    sent_json = {
                 "sent_id": col_sent.id,
                 "text": col_sent.text,
                 "opinions": []
                }
    # assign character offsets to each token
    i = 0
    for token in col_sent.tokens:
        j = i + len(token.form)
        token.char_offsets = (i, j)
        assert col_sent.text[i:j] == token.form, "{} {}:{}".format(print(col_sent, i, j))
        i = j + 1

    # find all roots, i.e. 0:exp-(Positive|Neutral|Negative)
    roots = find_roots(col_sent)

    # gather any other tokens belonging to sentiment expressions
    expressions = gather_expressions(roots, col_sent)

    # get polarities
    polarities = get_polarities(roots)

    # get targets corresponding to sentiment expression
    targets = gather_targets(roots, col_sent)

    # get holders corresponding to sentiment expression
    holders = gather_holders(roots, col_sent)

    assert len(expressions) == len(polarities) == len(targets) == len(holders)

    # put these into opinion dictionaries
    for i, root in enumerate(roots):
        """
        opinion = {"Source": [[], []],
                   "Target": [[], []],
                   "Polar_expression": [[], []],
                   "Polarity": "",
                   "Intensity": "average"
                   }
        """
        for targ in targets[i]:
            for holder in holders[i]:
                opinion = {"Source": holder,
                           "Target": targ,
                           "Polar_expression": expressions[i],
                           "Polarity": polarities[i],
                           "Intensity": "Standard"
                           }
                sent_json["opinions"].append(opinion)
    return sent_json

def convert_conllu_to_json(conllu_sents):
    return [convert_col_sent_to_json(sent) for sent in conllu_sents]
    #for i, sent in enumerate(conllu_sents):
    #    try:
    #        convert_col_sent_to_json(sent)
    #    except:
    #        print(i)


def main():
    """
    Converts the conllu format to sentiment graph jsons
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="the conllu file to be converted to json")
    parser.add_argument("outfile", help="the output json file")
    args = parser.parse_args()

    sentences = list(read_col_data(args.infile))

    json_sentences = convert_conllu_to_json(sentences)

    with open(args.outfile, "w") as outfile:
        json.dump(json_sentences, outfile)


if __name__ == "__main__":
    main()
