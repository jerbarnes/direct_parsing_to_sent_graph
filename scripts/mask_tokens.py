import json
import numpy as np
import re
import random
import itertools
import argparse 

from tqdm import tqdm
from time import time
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", type=str, required=True, help="path to train data json")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="output json (NB: is overwritten)")
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


def process_text_entry(text_entry, id_suffix, mask_token, token_id_to_mask=None):
    res_entry = {}
    res_entry["sent_id"] = text_entry["sent_id"] + id_suffix
    text = text_entry["text"]
    opinions =  json.loads(json.dumps(text_entry["opinions"]))
    
    opinion_masked_text = temporary_mask_expressions(text, opinions)
    tokens_with_separators = re.split(r'(\W+)', opinion_masked_text)
    tokens = tokens_with_separators[::2]

    n_tokens = len(tokens)
    n_tokens_to_mask = 1
    if n_tokens_to_mask == 0:
        return None, None, None
    if token_id_to_mask is None:
        token_ids_to_mask = random.sample(range(n_tokens), n_tokens_to_mask)
        token_ids_to_mask = [i*2 for i in token_ids_to_mask]
    else:
        token_ids_to_mask = [token_id_to_mask]
    if token_ids_to_mask[0] >= len(tokens_with_separators):
        return None, None, None
    res_token_list = []
    pos = 0
    min_token_len = 3
    mask_token_len = len(mask_token)
    success = False
    token_to_mask = ''
    for token_sep_id, token in enumerate(tokens_with_separators):
        token_len = len(token)
        if token_sep_id % 2 == 0 and token_len >= min_token_len and token_sep_id in token_ids_to_mask:
            success = True
            token_to_mask = text[pos:pos+token_len]
            res_token_list.append(mask_token)
            offset = mask_token_len - token_len
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
    if not success:
        return None, None, None
    res_text = ''.join(res_token_list)

    res_entry["text"] = res_text
    res_entry["opinions"] = opinions

    return res_entry, token_to_mask, token_ids_to_mask[0]


def generate_augmented_text_entries(text_entry, unmasker, threshold):
    res_entries = []
    for token_id_to_mask in range(100):
        masked_entry, masked_token, masked_token_id = process_text_entry(
            text_entry,
            id_suffix="_MASKED",
            mask_token='<mask>',
            token_id_to_mask=token_id_to_mask
        )
        if masked_entry is None:
            continue
        assert masked_token_id == token_id_to_mask

        unmasked = unmasker(masked_entry['text'])
        unmasked = [d for d in unmasked if d['score'] >= threshold and d['token_str'] != masked_token]



        for i, d in enumerate(unmasked):
            unmasked_entry, _, __ = process_text_entry(
                text_entry,
                id_suffix=f"_MASKED_{i}",
                mask_token=d['token_str'],
                token_id_to_mask=masked_token_id
            )
            res_entries.append(unmasked_entry)
    return res_entries


def main():
    # random.seed(10123)
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")
    unmasker = pipeline('fill-mask', model='xlm-roberta-large')

    with open(args.input_json) as inp:
        original_data = json.load(inp)
    print(f"original data len: {len(original_data)}")


    augmented_data = []
    for entry_num, entry in tqdm(list(enumerate(original_data))):
        augmented_data.append(generate_augmented_text_entries(entry, unmasker, threshold=0.5))
    augmented_data = list(itertools.chain(*augmented_data))
    print(f"augmented data len: {len(augmented_data)}")


    for d in augmented_data:
        for o in d['opinions']:
            e_texts, e_spans  = o['Polar_expression']
            for e_text, e_span in zip(e_texts, e_spans):
                e_start, e_end = map(int, e_span.split(':'))
                assert e_text == d['text'][e_start:e_end]

    combined_data = original_data + augmented_data
    with open(args.output_file, 'w') as otp:
        json.dump(combined_data, otp)

if __name__ == "__main__":

    main()
