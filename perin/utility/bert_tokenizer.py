#!/usr/bin/env python3
# coding=utf-8

def fix_quotes(text, quote_symbol='"'):
    n_quotes = text.count(f" {quote_symbol}") + text.count(f"{quote_symbol} ") - text.count(f" {quote_symbol} ")
    if (
        n_quotes == 0
        or (n_quotes % 2) == 1
        or f"{quote_symbol}{quote_symbol}" in text
        or f"{quote_symbol} {quote_symbol}" in text
    ):
        return text

    i, i_quote, n_changes = 0, 0, 0
    while i < len(text):
        if text[i] != quote_symbol or (i - 1 >= 0 and text[i - 1] != ' ' and i + 1 < len(text) and text[i + 1] != ' '):
            i += 1
            continue

        if (i_quote % 2) == 0:
            if i > 0 and text[i - 1] != ' ':
                text = text[:i] + ' ' + text[i:]
                i += 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1] == ' ':
                text = text[:i + 1] + text[i + 2:]
                n_changes += 1
        else:
            if i > 0 and text[i - 1] == ' ':
                text = text[:i - 1] + text[i:]
                i -= 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1].isalnum():
                text = text[:i + 1] + ' ' + text[i + 1:]
                n_changes += 1

        i_quote += 1
        i += 1

    return text


def detokenize(tokens, compact_dashes=False):
    text = ' '.join(tokens)
    # text = normalize_abbreviations(text)

    if compact_dashes:
        text = text.replace(' - ', '-')

    for i in range(len(text) - 2, -1, -1):
        if text[i] == '.' and (text[i + 1].isupper() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['?', '!', '…', '’'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ',' and (text[i + 1].isalpha() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in [';', ')', ']', '}', '%'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ':' and (text[i + 1] in ['‘', '(', '[', '{'] or (text[i + 1].isalnum() and (not text[i + 1].isnumeric() or i - 1 < 0 or not text[i - 1].isnumeric()))):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['(', '[', '{'] and text[i + 1] == ' ':
            text = text[:i+1] + text[i+2:]
        elif text[i] == ' ' and text[i+1] in ['.', ';', ':', '?', '!', '…', ',', '’', ')', ']']:
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1] in ['$', '£', '€'] and text[i + 1].isnumeric():
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1].isnumeric() and text[i + 1] == '%':
            text = text[:i] + text[i+1:]

    text = fix_quotes(text, '"')
    text = fix_quotes(text, "'")

    spans = []
    word_offset, char_offset = 0, 0
    for i, ch in enumerate(text):
        if ch == ' ':
            if tokens[word_offset][char_offset] == ' ':
                char_offset += 1
            continue

        assert ch == tokens[word_offset][char_offset], f"{text}\n{' '.join(tokens)}\n{tokens[word_offset]}\n{char_offset} {ch}"

        if char_offset == 0:
            start = i

        if char_offset == len(tokens[word_offset]) - 1:
            end = i + 1
            spans.append((start, end))
            word_offset += 1
            char_offset = 0
        else:
            char_offset += 1

    return text, spans


def bert_tokenizer(example, tokenizer, encoder):
    if "xlm" in encoder.lower():
        separator = '▁'
    elif "roberta" in encoder.lower():
        separator = 'Ġ'
    elif "bert" in encoder.lower():
        separator = '##'
    else:
        raise Exception(f"Unsupported tokenization for {encoder}")

    sentence, _ = detokenize(example["input"])
    original_tokens = [''.join([t.lstrip(separator).lower().strip() for t in tokenizer.tokenize(token)]) for token in example["input"]]
    tokenized_tokens = [token.lstrip(separator).lower().strip() for token in tokenizer.tokenize(sentence)]

    to_scatter, to_gather, to_delete = [], [], []
    orig_i, orig_offset, chain_length = 0, 0, 0
    unk_roll = False

    for i, token in enumerate(tokenized_tokens):
        chain_length += 1

        while orig_i < len(original_tokens) - 1 and orig_offset >= len(original_tokens[orig_i]):
            orig_i, orig_offset = orig_i + 1, 0
            chain_length = 0

        if token in ["<unk>", "[unk]", "[UNK]"]:
            unk_roll = True
            to_gather.append(i + 1)
            to_scatter.append(orig_i)
            if chain_length > 5:
                to_delete.append(i)
            continue

        elif unk_roll:
            found = False
            for orig_i in range(orig_i, len(original_tokens)):
                for orig_offset in range(len(original_tokens[orig_i])):
                    original_token = original_tokens[orig_i][orig_offset:]
                    if original_token.startswith(token) or token.startswith(original_token):
                        chain_length = 0
                        found = True
                        break
                if found:
                    break

        original_token = original_tokens[orig_i][orig_offset:]
        unk_roll = False

        if original_token.startswith(token):
            to_gather.append(i + 1)
            to_scatter.append(orig_i)
            orig_offset += len(token)
            if chain_length > 5:
                to_delete.append(i)
            continue

        # print(f"BERT parsing error in sentence {example['id']}: {example['sentence']}")

        unk_roll = True
        to_gather.append(i + 1)
        to_scatter.append(orig_i)

    bert_input = tokenizer.encode(sentence, add_special_tokens=True)
    to_gather, to_scatter, bert_input = reduce_bert_input(to_gather, to_scatter, bert_input, to_delete)

    return to_scatter, bert_input


def reduce_bert_input(to_gather, to_scatter, bert_input, to_delete):
    new_gather, new_scatter = [], []
    offset = 0
    for i in range(len(to_gather)):
        if to_gather[i] - 1 in to_delete:
            offset += 1
        else:
            new_gather.append(to_gather[i] - offset)
            new_scatter.append(to_scatter[i])
    bert_input = [w for i, w in enumerate(bert_input) if i - 1 not in to_delete]
    return new_gather, new_scatter, bert_input
