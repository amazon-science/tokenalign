# flake8: noqa
"""
Helper function for constrained generation
"""
from typing import List, Optional

import pygtrie
import torch

SPIECE_UNDERLINE = "▁"


def convert_tokens_to_string_utf8(tokenizer, tokens, is_llama=False):
    """Converts a sequence of tokens (strings for sub-words) in a single string."""
    # import pdb; pdb.set_trace()
    text = "".join(tokens)
    if text.find(" ") != -1:
        raise ValueError("Find whitespaces in converted token strings")
    ## for llama is_llama is True, True for other tokenizers
    if not is_llama:
        text = text.replace(SPIECE_UNDERLINE, "")
    else:
        # llama tokenizer decode will remove the first white space if exists
        # llama tokenizer has problem with newline decoding
        text = text.replace("<0x0A>", "\n")
    return text


def no_underline_prefix(text):
    if text.startswith(SPIECE_UNDERLINE):
        return text.replace(SPIECE_UNDERLINE, "")
    else:
        return text


def get_allowed_token_list(prompt_suffix: str, vocab_trie: pygtrie.Trie):
    """
    Given a prompt suffix, the allowed list consisting two parts:
    1. Tokens that have the prompt_suffix as its own prefix,
       e.g. given prompt suffix G, Gy/Ga and every starts with G, would count
    2. Tokens are that prefixes of current prompt suffix,
       e.g. given prompt suffix Game, ['', 'G', 'Ga', 'Gam'] are prefixes

    Combining those 2 sets would be all allowed tokens we can select next tokens
    that matches prompt_suffix
    """
    context_as_prefix_list, item_is_prefix_list = [], []
    try:
        # return al items that has prompt_suffix as its prefix
        # for example G, will give you G/Gy/Gz/Graphics, etc
        context_as_prefix_list = [
            y for y in vocab_trie.iteritems(prompt_suffix, shallow=False)
        ]
    except:
        pass
    try:
        # return all items where the token is part prefix of the prompt
        # for example, GGG, will give you '', 'G', 'GG', 'GGG'
        item_is_prefix_list = [y for y in vocab_trie.prefixes(prompt_suffix)]
    except:
        pass
    # import pdb; pdb.set_trace()
    allowed_tokens_idxs = context_as_prefix_list + item_is_prefix_list
    allowed_tokens_idxs = torch.cat([idxs[1] for idxs in allowed_tokens_idxs])
    return allowed_tokens_idxs


def apply_allowed_token_mask(
    prompt_suffix,
    vocab_trie,
    vocab,
    next_token_scores=None,
    single_space_mask=None,
    mask_empty_string=True,
    limit_to_long_spaces=False,
):
    # BenA -- this works for starcoder but we should make it automated
    ENCODED_SPACE = "Ġ"
    ENCODED_TAB = "ĉ"
    ENCODED_NEWLINE = "Ċ"
    MAX_SPACES_SP_TOKEN = 16
    MAX_TABS_SP_TOKEN = 5

    if prompt_suffix is None:
        raise ValueError("Prompt suffix can't be None")
    # modify the tensor in place if it's provided
    # 1. single space case (most common). without caching it can take 30ms
    if (
        prompt_suffix == ENCODED_SPACE
        and next_token_scores is not None
        and single_space_mask is not None
    ):
        if next_token_scores.device != single_space_mask.device:
            single_space_mask = single_space_mask.to(next_token_scores.device)  # 6 ms
        next_token_scores += single_space_mask
        return
    # 2. construct token list on the fly
    prompt_suffix_char_set = set(prompt_suffix)
    if limit_to_long_spaces and prompt_suffix_char_set in [
        {ENCODED_SPACE},
        {ENCODED_TAB},
        {ENCODED_NEWLINE},
    ]:
        # this is the case where prompt suffix contains all spaces, tabs, or newlines
        # this corresponds to only 1 token since backtracking for special tokens is only 1 token
        # we also want to limit CG to match only long spaces (X-1, X, ..., max number of spaces in special token)
        # this step particularly does not allow shorter spaces than N-1 (which would be out of distribution per tokenization)
        if prompt_suffix_char_set == {ENCODED_SPACE}:
            # include only X-1, X, X+1, .. spaces
            allowed_tokens = [
                ENCODED_SPACE * num_space
                for num_space in range(len(prompt_suffix) - 1, MAX_SPACES_SP_TOKEN + 1)
            ]
        elif prompt_suffix_char_set == {ENCODED_TAB}:
            # include only X, X+1, .. tabs
            allowed_tokens = [
                ENCODED_TAB * num_tab
                for num_tab in range(len(prompt_suffix), MAX_TABS_SP_TOKEN + 1)
            ]
        else:
            allowed_tokens = [ENCODED_NEWLINE, ENCODED_NEWLINE * 2]

        allowed_tokens_idxes = torch.LongTensor(
            [vocab[token] for token in allowed_tokens]
        )
    else:
        allowed_tokens_idxes = get_allowed_token_list(prompt_suffix, vocab_trie)

    if next_token_scores is not None:

        if len(next_token_scores.size()) != 1:
            raise ValueError("Expecting to match with 1 token")
        mask = (
            torch.Tensor(len(next_token_scores))
            .fill_(float("-inf"))
            .to(next_token_scores.device)
        )
        mask[allowed_tokens_idxes] = 0.0
        # TODO -- mask empty string -- see if there's any empty string
        #if mask_empty_string:
        #    mask[51030] = float(
        #        "-inf"
        #    )  # this token is empty string -- need to mask it out as well
        next_token_scores += mask


def get_pretokenized_position_from_ids(ids, tokenizer, min_position=-3, is_llama=False):
    for _idx in range(len(ids)):
        idx = -(_idx + 1)
        if idx == min_position:
            return idx  # cover edge case of unusual strings
        token = tokenizer.convert_ids_to_tokens([ids[idx]])[0]
        # if llama tokenizer: is_llama is True, otherwise False
        if not is_llama and token.startswith(SPIECE_UNDERLINE):
            return idx
        elif token in tokenizer._additional_special_tokens:
            # this will cover all Ġ cases
            return idx
    return -1  # fallback


def build_vocab_trie(tokenizer, vocab, is_llama=False):
    vocab_trie = pygtrie.CharTrie()
    for word in vocab:
        if not is_llama:
            # regular tokenizer
            _word = no_underline_prefix(word)
        else:
            # # llama
            # SPIECE_UNDERLINE+"return" is in vocab, " "+"return" is not
            # '<0x0A>' in vocab but '\n' not in vocab
            # llama tokenizer is not lossless: 
            # tokenizer.decode(tokenizer.convert_tokens_to_ids('\n')) = '<unk>'
            # tokenizer.decode(tokenizer.convert_tokens_to_ids('<0x0A>')) = '<unk>'
            # tokenizer.decode(tokenizer.encode('\n')) = '<s>\n'
            # if word == "<0x0A>":
            #     import pdb; pdb.set_trace()
            _word = word
            _word = _word.replace("<0x0A>", "\n")
            
        if _word in vocab_trie:
            vocab_trie[_word] = torch.cat(
                [vocab_trie[_word], torch.LongTensor([1]).fill_(vocab[word])]
            )
        else:
            vocab_trie[_word] = torch.LongTensor([1]).fill_(vocab[word])
    return vocab_trie


def prepare_vocab_trie(tokenizer, device=None):
    # 1. prepare vocab trie
    tokenizer.vocab = tokenizer.get_vocab()
    vocab_trie = build_vocab_trie(tokenizer, tokenizer.vocab, is_llama=tokenizer.is_llama)
    tokenizer.vocab_trie = vocab_trie
    # 2. cache single space mask
    single_space_idxs = torch.cat(
        [
            idxs[1]
            for idxs in [y for y in tokenizer.vocab_trie.iteritems("Ġ", shallow=False)]
        ]
    )
    single_space_mask = torch.Tensor(len(tokenizer)).fill_(float("-inf"))
    single_space_mask[single_space_idxs] = 0.0
    tokenizer.single_space_mask = single_space_mask
    if device is not None:
        tokenizer.single_space_mask = tokenizer.single_space_mask.to(device)
