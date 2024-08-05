import re
from typing import List, Optional, Tuple

import numpy
import torch
try:
    from transformers.generation.stopping_criteria import (
        MaxLengthCriteria,
        MaxTimeCriteria,
        StoppingCriteria,
        StoppingCriteriaList,
    )
except ImportError:
    from transformers.generation_stopping_criteria import (
        MaxLengthCriteria,
        MaxTimeCriteria,
        StoppingCriteria,
        StoppingCriteriaList,
    )


USER_DEFINED_SYMBOLS = [
    "ĉ",
    "ĉĉ",
    "ĉĉĉ",
    "ĉĉĉĉ",
    "ĉĉĉĉĉ",
    "Ċ",
    "ĊĊ",
    "čĊ",
    "čĊčĊ",
    "ĠĠĠ",
    "ĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠ",
    "ĠĠĠĠ",
    "ĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
]

SCOPE_COMPLETION = "scope_completion"
LINE_COMPLETION = "line_completion"


TOKEN_NEW_LINE_DICT = {"\n": 1, "\n\n": 2, "\r\n": 1, "\r\n\r\n": 2}
TOKEN_INDENT_LEVEL_SET = {
    " ",
    " " * 3,
    " " * 5,
    " " * 7,
    " " * 9,
    " " * 11,
    " " * 13,
    " " * 15,
    " " * 2,
    " " * 4,
    " " * 6,
    " " * 8,
    " " * 10,
    " " * 12,
    " " * 14,
    " " * 16,
    "\t",
    "\t\t",
    "\t" * 3,
    "\t" * 4,
    "\t" * 5,
}


class StoppingCriteriaPython(StoppingCriteria):
    def __init__(
        self,
        eog_type,
        max_lines,
        batch_size,
        tok,
        input_length,
        outer_indent,
        init_input_ids,
    ):
        self.max_lines = max_lines
        self.eog_type = eog_type
        self.tok = tok
        self.batch_size = batch_size
        self.prefixes = [""] * batch_size
        self.input_length = input_length
        self.num_new_lines = [0] * batch_size
        self.indents = [0] * batch_size
        self.outer_indent = outer_indent
        self.finished = [0] * batch_size
        self.at_first_line = [True] * batch_size
        self.token_size_at_finish = [0] * batch_size
        # handle constrained generation -- keep track of prompt suffix
        self.init_input_ids = init_input_ids
        self.original_prompt_len = len(init_input_ids[0])
        self.len_with_max_backtrack = len(init_input_ids[0]) - 4
        prompt_suffix_ids = init_input_ids[:, self.len_with_max_backtrack :]
        self.prompt_suffix_list = self.tok.batch_decode(
            prompt_suffix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.constrained_generation_on = [True] * batch_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -1]
        if self.eog_type == LINE_COMPLETION:
            for idx in range(input_ids.size(0)):
                if self.finished[idx] == 0:
                    token = get_token_minus_prompt(self, input_ids, idx)
                    self.token_size_at_finish[idx] += 1
                    self.prefixes[idx] += token
                    if self.num_new_lines[idx] < self.max_lines:
                        self.num_new_lines[idx] += count_new_lines(token)
                    if self.num_new_lines[idx] >= self.max_lines:
                        self.finished[idx] = 1
        else:
            # TODO: consider to only trigger EoS detection
            #  when there is a complete line generated?
            for idx in range(input_ids.size(0)):
                token = self.tok.decode(last_ids[idx])
                self.token_size_at_finish[idx] += 1
                self.prefixes[idx] += token
                if self.finished[idx] == 0:
                    if self.prefixes[idx][-1] == "\n":
                        self.indents[idx] = 0
                        self.at_first_line[idx] = False
                    elif (
                        re.match("\S", self.prefixes[idx][-1])  # noqa: W605
                        and not self.at_first_line[idx]
                    ):
                        if self.indents[idx] >= 0:
                            curr_line_indent = (
                                self.indents[idx] + 1
                                if token.find(" ") == 0
                                else self.indents[idx]
                            )
                            if curr_line_indent <= self.outer_indent:
                                self.finished[idx] = 1
                                self.token_size_at_finish[idx] -= 1
                        self.indents[idx] = -1
                    elif token in TOKEN_INDENT_LEVEL_SET and self.indents[idx] >= 0:
                        self.indents[idx] += len(token)
        return sum(self.finished) == self.batch_size


def count_brace_balance(v):
    return v.count("{") - v.count("}")


def count_new_lines(token):
    return token.count("\n")


def get_token_minus_prompt(
    self,
    input_ids_plus_gen,
    idx,
):
    """
    For constrained generation, if the last token in prompt is 'Sys',
    and the model generate a full token 'System', then partial token will be 'tem'.
    This method returns the freshly generated characters (token string)

    Corner case that can effect end of scope: prompt ends with '{'.
    If the constrained generation replaces this with a token representing '{{'
    then we want to count the braces properly starting only from the second {
    in this case, partial_generation will be '{' only, not a full token {{

    Or if the constrained generation generates '{', then the fresh now token string is ''
    (empty string). this empty string will be used to perform braces matching, new line counting, etc.
    By doing this, all end of generation logic will start only with the freshly generated characters.
    """
    if (
        input_ids_plus_gen.size(0) > self.init_input_ids.size(0)
        or not self.constrained_generation_on[idx]
    ):
        # return normal token if the constrained generation mode is already off
        # if the broadcasting already happens, this means the constrained generation mode is already off
        token = self.tok.decode(
            input_ids_plus_gen[idx, -1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    else:
        prompt_suffix_plus_generate = self.tok.decode(
            input_ids_plus_gen[idx, self.len_with_max_backtrack :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prompt_suffix = self.prompt_suffix_list[idx]
        if not prompt_suffix_plus_generate.startswith(prompt_suffix):
            # in this case, the current generation is still matching with the prompt
            # the stopping criteria will not do brace counting or newline matching, etc.
            # because we set the token to empty string
            token = ""
        else:
            # after this token, constrained generation is off
            self.constrained_generation_on[idx] = False
            new_generation = prompt_suffix_plus_generate[len(prompt_suffix) :]
            if new_generation != "":
                token = new_generation
            else:
                token = ""
    return token


class StoppingCriteriaJavaJs(StoppingCriteria):
    def __init__(
        self,
        eog_type,
        max_lines,
        batch_size,
        tok,
        brace_balance_voc_dict,
        init_input_ids,
    ):
        self.eog_type = eog_type
        self.max_lines = max_lines
        self.batch_size = batch_size
        self.prefixes = [""] * batch_size
        self.brace_balance = numpy.zeros(batch_size)
        self.num_new_lines = [0] * batch_size
        self.tok = tok
        self.brace_balance_voc_dict = brace_balance_voc_dict
        self.finished = [0] * batch_size
        self.token_size_at_finish = [0] * batch_size
        # handle constrained generation -- keep track of prompt suffix
        self.init_input_ids = init_input_ids
        self.original_prompt_len = len(init_input_ids[0])
        self.len_with_max_backtrack = len(init_input_ids[0]) - 4
        prompt_suffix_ids = init_input_ids[:, self.len_with_max_backtrack :]
        self.prompt_suffix_list = self.tok.batch_decode(
            prompt_suffix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.constrained_generation_on = [True] * batch_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # note: input_ids really mean input + generated tokens
        if self.eog_type == LINE_COMPLETION:
            for idx in range(input_ids.size(0)):
                if self.finished[idx] == 0:
                    token = get_token_minus_prompt(self, input_ids, idx)
                    self.token_size_at_finish[idx] += 1
                    self.prefixes[idx] += token
                    if self.num_new_lines[idx] < self.max_lines:
                        self.num_new_lines[idx] += count_new_lines(token)
                    if self.num_new_lines[idx] >= self.max_lines:
                        self.finished[idx] = 1
            return sum(self.finished) == self.batch_size
        else:
            last_ids = input_ids[:, -1]
            for idx, wid in enumerate(last_ids):
                token = get_token_minus_prompt(self, input_ids, idx)
                # replace brace counting by id lookup with counting in string directly
                brace_count = count_brace_balance(token)
                if self.brace_balance[idx] >= 0:
                    self.token_size_at_finish[idx] += 1
                if brace_count != 0 and self.brace_balance[idx] >= 0:
                    self.brace_balance[idx] += brace_count

            return sum(self.brace_balance < 0) == self.batch_size


def get_stopping_criteria(
    start_length: int,
    max_new_tokens: int,
    max_time: float,
    customized_criterion: Optional[StoppingCriteria],
) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    # Note the order here matters
    # Custom stopping criteria should come first
    if customized_criterion:
        stopping_criteria.append(customized_criterion)
    # THIS MAY CHANGE IN FUTURE HF VERSIONS
    # if max_new_tokens is not None:
    #     """FROM THE DOCS
    #     "The class `MaxNewTokensCriteria` is deprecated. "
    #     f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
    #     "with `max_length = start_length + max_new_tokens` instead.",
    #     """
    #     print(f"Length based stopping criteria is automatically added in generate py through generation_config")
    #     # stopping_criteria.append(
    #     #     MaxLengthCriteria(
    #     #         max_length=start_length+max_new_tokens
    #     #     )
    #     # )
    if max_time is not None:
        stopping_criteria.append(MaxTimeCriteria(max_time=max_time))

    return stopping_criteria


def get_stopping_criteria_per_language(
    language,
    eog_type,
    max_lines,
    num_return_sequences,
    input_len,
    input_indent,
    tokenizer,
    max_new_tokens,
    init_input_ids,
):
    bracket_based_languages = [
        "java",
        "javascript",
        "typescript",
        "kotlin",
        "php",
        "rust",
        "cpp",
        "csharp",
        "go",
        "swift",
        "scala",
        "perl"
    ]
    python_like_languages = ["python", "ruby"]
    supported = language in bracket_based_languages + python_like_languages
    assert supported, f"Language {language} not supported"

    # bracket based
    if language in bracket_based_languages:
        vocab_dict = tokenizer.get_vocab()  # obtain dict from cache (saves 20 ms)
        voc_brace_balance_ids = dict(
            [
                (vocab_dict[v], v.count("{") - v.count("}"))
                for v in vocab_dict.keys()
                if v.count("}") > 0 or v.count("{") > 0
            ]
        )
        customized_criterion = StoppingCriteriaJavaJs(
            eog_type,
            max_lines,
            num_return_sequences,
            tokenizer,
            voc_brace_balance_ids,
            init_input_ids,
        )
    else:
        # ruby and python
        customized_criterion = StoppingCriteriaPython(
            eog_type,
            max_lines,
            num_return_sequences,
            tokenizer,
            input_len,
            input_indent,
            init_input_ids,
        )

    stopping_criteria = get_stopping_criteria(
        start_length=input_len,
        max_new_tokens=max_new_tokens,  # max_length - input_ids.shape[1],
        max_time=100000,  # high max time
        customized_criterion=customized_criterion,
    )

    return stopping_criteria
