import ast


def is_valid_python(code):
    try:
        try:
            pared_code = ast.parse(code)
        except SyntaxError:
            return False
    except Exception as e:
        print("Exception: ", e)
        return False
    return pared_code


def get_function_from_ast(parsed_code, code, option="func_ast_last"):
    """
    Author: Ben
    """
    # This grabs the entire generation up til the last function that is valid
    # this is still a greedy approach at function level
    # another approach is to grab up to the first function
    # note: there can be import statements, variables, before first function

    # print(f"Starting from the following code\n{code}\n\n\n")
    # 1. get the last function
    assert option in [
        "func_ast_first",
        "func_ast_last",
    ], f"Invalid post process option {option}"
    for i in range(len(parsed_code.body)):
        idx = -i - 1 if option == "func_ast_last" else i
        if type(parsed_code.body[idx]) == ast.FunctionDef:
            break
        idx = None
    assert idx is not None, "No function found"
    function_segment = ast.get_source_segment(code, parsed_code.body[idx])
    # print(f"Found function segment at idx = {-i-1}\n{function_segment}\n\n\n")
    position = code.find(function_segment)
    # function_segment = code[position: position+len(function_segment)]
    function_segment_plus_previous = code[: position + len(function_segment)]
    return function_segment_plus_previous


def filter_func_ast(codestr, init_str, tokenizer):
    """
    Author: Ben
    """
    # len_init_str = len(tokenizer.encode(init_str))
    additional_gen_str = codestr[len(init_str) :]
    additional_gen_toks = tokenizer.encode(additional_gen_str)
    for position in range(len(additional_gen_toks) - 1, -1, -1):
        generation_up_to_position_toks = additional_gen_toks[:position]
        generation_to_to_position_str = tokenizer.decode(
            generation_up_to_position_toks, clean_up_tokenization_spaces=False
        )
        parsed_code = is_valid_python(init_str + generation_to_to_position_str)
        if parsed_code:
            # print(
            #    f"valid at position {position} / {len(additional_gen_toks) - 1}. init str len {len_init_str}"
            # )
            function_segment_plus_previous = get_function_from_ast(
                parsed_code, init_str + generation_to_to_position_str
            )
            return function_segment_plus_previous[len(init_str) :]
    print("Warning - no valid substring")
    return codestr  # if nothing is valid


def get_token_position_by_string(target_str, outputs, tokenizer, skip_special_tokens):
    for position in range(1, len(outputs) + 1):
        gen_str = tokenizer.decode(
            outputs[:position],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        if gen_str.rstrip() == target_str.rstrip():
            return position  # not including outputs[position]
        if gen_str.startswith(target_str) and target_str != "":
            print("Cannot find an exact match, use approx!")
            print(f"output length: {len(outputs)}")
            print(target_str)
            print("-----------------------")
            print(gen_str)
            return position
    if target_str.rstrip() == "":
        if target_str == "":
            print("generated empty string!")
        else:
            print("generated only white space!")
        return 0
    print(f"output length: {len(outputs)}")
    print(target_str)
    print("-----------------------")
    print(gen_str)
    raise RuntimeError("Cannot match prefix returned by AST.")



def filter_valid_code(
    true_str_input,
    execution_prompt,
    inputs,
    sequences,
    initial_context_length,
    tokenizer,
    task_id=None,
    has_special_tokens=False,
    post_process="greedy",
    replace_unk=False,
    skip_special_tokens=True,
    mean_logp=None,
    use_language_tag=0,
):
    samples = []

    """
    Due to tokenizer non lossless-ness, the decoded original prompt and
    the real original prompt are not the same.

    Due to constrained generation, input tokens not not necessarily match
    with the new input tokens (but match by characters instead)
    """
    # decoded string of the original prompts
    decoded_original_prompt = tokenizer.batch_decode(
        inputs[:, use_language_tag:],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    # decoded string of the generated sequences including original prompt + generations
    # due to constrained generation, last few tokens of original prompt might be merged with first tokens of generations
    # therefore we need to string level matching instead of token level matching to extract the generations from sequences
    decoded_sequences = tokenizer.batch_decode(
        sequences[:, use_language_tag:],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    if execution_prompt is not None: 
        # usually execution prompt is the prompt alone without addons like fewshot examples
        decoded_execution_prompt = tokenizer.decode(
                tokenizer.encode(execution_prompt)[use_language_tag:],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )

    for sample_id, (decoded_seq, decoded_input) in enumerate(zip(decoded_sequences, decoded_original_prompt)):
        if decoded_seq[:len(decoded_input)] != decoded_input: import pdb; pdb.set_trace()
        assert decoded_seq[:len(decoded_input)] == decoded_input, f"decoded sequences original prompt not matching decoded input, wrong tokenization decoding! \n||{decoded_input}||\n\n->\n\n||{decoded_seq[:len(decoded_input)]}||"
        decoded_output = decoded_seq[len(decoded_input):]
        outputs = tokenizer.encode(decoded_output)
        
        processed_prompt = decoded_input

        if execution_prompt is not None:
            processed_execution_prompt = decoded_execution_prompt
        else:
            processed_execution_prompt = decoded_input

        is_valid = False
        for position in range(len(outputs), 0, -1):
            gen_up_to_pos_toks = outputs[:position]
            gen_up_to_pos_str = tokenizer.decode(
                gen_up_to_pos_toks,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = gen_up_to_pos_str
            code = (
                processed_execution_prompt + gen_up_to_pos_str
            )
            parsed_code = is_valid_python(code)
            # print(code)
            if parsed_code:
                is_valid = True
                # print(f"valid at position {position} / {len(outputs) - 1}. ")
                if post_process != "greedy":
                    try:
                        function_segment_plus_previous = get_function_from_ast(
                            parsed_code,
                            code,
                            option=post_process,
                        )
                        generated_part = function_segment_plus_previous[
                            len(processed_execution_prompt) :
                        ]
                    except Exception as e:
                        print("Something went wrong...", e)
                        generated_part = gen_up_to_pos_str
                elif post_process == "greedy":
                    generated_part = gen_up_to_pos_str
                else:
                    assert False, f"post processing method {post_process} not supported"

                if task_id is None:
                    return generated_part
                    # TODO -- when is this used?
                if mean_logp is None:
                    score = None
                else:
                    if post_process != "greedy":
                        position = get_token_position_by_string(
                            generated_part,
                            outputs,
                            tokenizer,
                            skip_special_tokens,
                        )
                    if position == 0:
                        score = -1e8
                    else:
                        score = mean_logp[sample_id][position - 1]

                samples.append(
                    dict(
                        task_id=task_id,
                        completion=generated_part,
                        ori_pred=origin_pred,
                        input=true_str_input,
                        mean_logp=score,
                    )
                )
                break
        if not is_valid:
            predictions = tokenizer.decode(
                outputs,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = predictions
            print("Warning - no valid substring")
            if task_id is None:
                return predictions
            samples.append(
                dict(
                    task_id=task_id,
                    completion=(processed_prompt + predictions)[
                        len(decoded_original_prompt) :
                    ],
                    ori_pred=(processed_prompt + origin_pred)[
                        len(decoded_original_prompt) :
                    ],
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )

    return samples


def inference_cut_off(
    true_str_input,
    inputs,
    sequences,
    token_len_prompt_input,
    tokenizer,
    skip_special_tokens,
    task_id,
    language,
    input_indent=0,
    mean_logp=None,
    java_class_completion=True,
):
    str_seqs = tokenizer.batch_decode(
        [seq for seq in sequences],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    str_input = tokenizer.batch_decode(
        inputs,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    results = []
    for sample_id, str_seq in enumerate(str_seqs):
        # # Add this as a temprorary fix if corner cases happened for consgen
        # if str_seq.find(str_input) != 0: 
        #     results.append(
        #         dict(
        #             task_id=task_id,
        #             completion=str_seq[len(str_input) :],
        #             ori_pred=str_seq[len(str_input) :],
        #             input=true_str_input,
        #             mean_logp=-1e8,
        #         )
        #     )
        #     continue
        assert (
            str_seq.find(str_input) == 0
        ), f"raw output = \n{str_seq}\n\n raw input = \n{str_input}"
        str_output = str_seq[len(str_input) :]
        # Complete function braces for brace-based languages
        # For Java, close with another brace for the class
        balance = 0
        if language in [
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
            "scala"
        ]:
            # count the balance for input
            for ch in str_input:
                if ch == "{": balance += 1
                if ch == "}": balance -= 1

            i = 0
            for i in range(len(str_output)):
                if str_output[i] == "{":
                    balance += 1
                elif str_output[i] == "}":
                    balance -= 1
                if balance == 0:
                    break
            generated_part = str_output[: i + 1]
            end_token_position = get_token_position_by_string(
                str_input + generated_part,
                sequences[sample_id],
                tokenizer,
                skip_special_tokens,
            )
            if mean_logp is not None:
                score = mean_logp[sample_id][
                    end_token_position - 1 - token_len_prompt_input
                ]
            else:
                score = -1e8
            # if language == "java" and java_class_completion:
            #     generated_part += "\n}"
            results.append(
                dict(
                    task_id=task_id,
                    completion=generated_part,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=score,
                )
            )
            # print("balance (number of open braces) =", balance)

        elif language == "python":
            # Not used in offline evaluation
            output_lines = str_output.split("\n")
            cutoff_output = output_lines[0]
            for line in output_lines[1:]:
                if not line.strip():
                    cutoff_output += "\n" + line
                elif len(line) - len(line.lstrip()) <= input_indent:
                    cutoff_output += "\n" + line[: len(line) - len(line.lstrip())]
                    break
                else:
                    cutoff_output += "\n" + line
            results.append(
                dict(
                    task_id=task_id,
                    completion=cutoff_output,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )
        elif language == "ruby":
            output_lines = str_output.split("\n")
            cutoff_output = output_lines[0]
            for line in output_lines[1:]:
                if not line.strip():
                    cutoff_output += "\n" + line
                elif len(line) - len(line.lstrip()) <= input_indent:
                    cutoff_output += "\n" + line[: len(line) - len(line.lstrip())]
                    break
                else:
                    cutoff_output += "\n" + line
            cutoff_output += "\nend\n"
            results.append(
                dict(
                    task_id=task_id,
                    completion=cutoff_output,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )
        else:
            assert False, f"Language {language} unsupported"
    return results
