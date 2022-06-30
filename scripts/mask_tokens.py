import argparse
import os
import datetime
import json
import re
import random
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", type=str, required=True, help="path to train data json")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="output json (NB: is overwritten)")
    parser.add_argument("-n", "--n_masks", type=float, required=True, help="max number of tokens to mask in each text")
    args = parser.parse_args()
    return args

def temporary_mask_expressions(text, opinions):
    intervals_to_mask = []
    symbols = list(text)
    for opinion in opinions:
        for token_type in ["Source", "Target", "Polar_expression"]:
            opinion_tokens = opinion[token_type][0]
            opinion_token_positions = opinion[token_type][1]
            for opinion_token, opinion_token_position in zip(opinion_tokens, opinion_token_positions):
                interval_start, interval_end = map(int, opinion_token_position.split(':'))
                interval_len = interval_end - interval_start
                symbols[interval_start:interval_end] = [' '] * interval_len
        
    opinion_masked_text = ''.join(symbols)
    assert(len(text) == len(opinion_masked_text))
    return opinion_masked_text


def process_text_entry(text_entry, max_masked_tokens):
    res_entry = {}
    res_entry["sent_id"] = text_entry["sent_id"] + "_MASKED"
    text = text_entry["text"]
    opinions = text_entry["opinions"]
    
    opinion_masked_text = temporary_mask_expressions(text, opinions)
    tokens_with_separators = re.split(r'(\W+)', opinion_masked_text)
    tokens = tokens_with_separators[::2]

    n_tokens = len(tokens)
    n_tokens_to_mask = random.randint(0, min(n_tokens, max_masked_tokens))
    if n_tokens_to_mask == 0:
        return None
    token_ids_to_mask = random.sample(range(n_tokens), n_tokens_to_mask)
    token_ids_to_mask = [i*2 for i in token_ids_to_mask]
    
    res_token_list = []
    pos = 0
    min_token_len = 1
    masked_token = "[MASK]"
    masked_token_len = len(masked_token)
    for token_sep_id, token in enumerate(tokens_with_separators):
        token_len = len(token)
        if token_sep_id % 2 == 0 and token_len >= min_token_len and token_sep_id in token_ids_to_mask:
            res_token_list.append(masked_token)
            offset = masked_token_len - token_len
            if offset != 0:
                for opinion in opinions:
                    for token_type in ["Source", "Target", "Polar_expression"]:
                        opinion_token_positions = opinion[token_type][1]
                        shifted_opinion_positions = []
                        for opinion_token_position in opinion_token_positions:
                            interval_start, interval_end = map(int, opinion_token_position.split(':'))
                            if interval_start > pos:
                                interval_start += offset
                                interval_end += offset
                            shifted_opinion_positions.append(f"{interval_start}:{interval_end}")
                        opinion[token_type][1] = shifted_opinion_positions
        else:
            res_token_list.append(text[pos:pos+token_len])
        pos += token_len
    res_text = ''.join(res_token_list)

    res_entry["text"] = res_text
    res_entry["opinions"] = json.loads(json.dumps(opinions))

    # print(f"orig: {text}\nmask: {opinion_masked_text}\nres:  {res_text}\n")
    # assert(len(text) == len(res_text))
    return res_entry


def main():
    # random.seed(10123)
    args = parse_arguments()

    with open(args.input_json) as inp:
        data = json.load(inp)

    masked_data = []
    for e_id, text_entry in enumerate(tqdm(data)):
        res_entry = process_text_entry(text_entry, args.n_masks)
        if res_entry is None:
            continue
        masked_data.append(res_entry)

    augmented_data =  data + masked_data

    with open(args.output_file, 'w') as otp:
        json.dump(augmented_data, otp)


if __name__ == "__main__":

    main()
