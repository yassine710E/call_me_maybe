# from Parser import Parser
# from FileLoader import FileLoader
# from llm_sdk.llm_sdk import Small_LLM_Model
# import json
# import os
# from itertools import zip_longest


# def isValid(s: str) -> bool:
#     if not s:
#         return False
#     new_s = str()
#     for ch in s:
#         if ch in ['{', '}']:
#             new_s += ch

#     stack = []
#     mapping = {"}": "{"}

#     for char in new_s:
#         if char in mapping:
#             top_element = stack.pop() if stack else '#'
#             if mapping[char] != top_element:
#                 return False
#         else:
#             stack.append(char)

#     return not stack


# def set_limited_tokens_for_args(functions_definition, target_name):
#     new_limited_tokens = []
#     list_params_names = [
#         param for f in functions_definition for param in f['parameters'].keys() if f['name'] == target_name]
#     for token in limited_tokens:
#         if token is None:
#             for i, param in enumerate(list_params_names):
#                 new_limited_tokens += model.encode(
#                     f"\"{param}\":")[0].tolist()
#                 new_limited_tokens.append(None)
#         else:
#             new_limited_tokens.append(token)
#     return new_limited_tokens


# def get_masked_logits(logits, next_expected_tokens) -> list[float] | None:

#     if isinstance(next_expected_tokens, int):
#         masked_logits = [score if index == next_expected_tokens else float(
#             '-inf') for index, score in enumerate(logits)]
#     elif isinstance(next_expected_tokens, list):
#         masked_logits = [score if index in next_expected_tokens else float(
#             '-inf') for index, score in enumerate(logits)]
#     elif next_expected_tokens is None:
#         masked_logits = logits
#     else:
#         masked_logits = None
#     return masked_logits


# def generate_token(vocabs, tokens, model,  limited_tokens):
#     i = 0
#     global limit_index_fn_name
#     function_name = str()
#     flag = True
#     answer = ""
#     auto_generation = False
#     while not isValid(answer):
#         logits = model.get_logits_from_input_ids(tokens)
#         n_logits = get_masked_logits(
#             logits, limited_tokens[i] if not auto_generation and i < len(limited_tokens) else None)
#         if n_logits is None:
#             return

#         if n_logits == logits and not auto_generation:
#             auto_generation = True

#         max_token = max(vocabs.keys(), key=lambda x: n_logits[x])
#         if i >= 3 and flag:
#             if function_name in [f['name'] for f in functions_definition]:
#                 i = limit_index_fn_name
#                 flag = False
#                 continue
#             else:
#                 function_name += model.decode([max_token])

#         elif not flag:
#             limited_tokens = set_limited_tokens_for_args(
#                 functions_definition, function_name)
#         answer += model.decode([max_token])
#         tokens.append(max_token)
#         os.system('cls' if os.name == 'nt' else 'clear')
#         print(model.decode(tokens))
#         if not auto_generation:
#             i += 1
#         if max_token in [1335, 2198] and auto_generation:
#             auto_generation = False
#     return answer


# def get_functions_data(functions, model):
#     names_of_functions = [f['name'] for f in functions]
#     two_d_tokens = []
#     for f_name in names_of_functions:
#         tensor_list_obj = model.encode(f_name)
#         tokens = tensor_list_obj[0].tolist()
#         two_d_tokens.append(tokens)

#     # reverse matrix (rows -> columns and columns -> ros)
#     reversed_two_d_tokens = [list(set(item for item in t if item))
#                              for t in zip_longest(*two_d_tokens)]

#     return reversed_two_d_tokens


# def get_limits_probabs(new_tokens, matrix_function_names):
#     global limit_index_fn_name
#     returned_val = list()
#     for index, id in enumerate(new_tokens):
#         if id == None:
#             returned_val = new_tokens[:index] + \
#                 matrix_function_names + new_tokens[index+1:]
#             limit_index_fn_name = index+len(matrix_function_names)
#             break
#     return returned_val


# def constrained_decoding(functions, model) -> list:
#     format_json = '{"name":"<|im_start|>fn<|im_end|>","args":{<|im_start|>args<|im_end|>}}'
#     tensor_list_obj = model.encode(format_json)
#     tokens = tensor_list_obj[0].tolist()
#     index = 0
#     new_tokens = list()
#     while index < len(tokens):
#         if tokens[index] == 151644:
#             new_tokens.append(None)
#             index += 3
#         else:
#             new_tokens.append(tokens[index])
#             index += 1

#     matrix_function_names = get_functions_data(functions, model)
#     new_tokens = get_limits_probabs(new_tokens, matrix_function_names)
#     return new_tokens


# def get_vocabs(model: Small_LLM_Model):
#     vocab_path = model.get_path_to_tokenizer_file()
#     with open(vocab_path, 'r', encoding="utf-8") as f:
#         vocabs = json.load(f)
#     raw_vocab = vocabs.get('model', {}).get('vocab', {})
#     return raw_vocab


# def build_sytem_prompt(functions):
#     lines = [
#         "STRICT SYSTEM RULE: You are a function-calling router. You NEVER answer the user directly.",
#         "",
#         "Task: choose exactly one function name from the list below that matches the user's intent,",
#         "and extract its arguments from the prompt.",
#         "",
#         "Rules:",
#         "1. If no function matches the user's intent, output {\"name\": \"none\", \"parameters\": {}}.",
#         "2. Never substitute a function for a different task, even if argument types match.",
#         "3. For parameters of type \"number\": ALWAYS include a decimal point in the value, even for",
#         "   whole numbers (write 3.0, not 3; write 1234567.89, not \"1234567.89\"). Never wrap a",
#         "   number in quotes.",
#         "4. For parameters of type \"string\": copy the exact substring from the user's prompt,",
#         "   character-for-character, including quotes, punctuation, casing, and backslashes.",
#         "   Do not paraphrase, reorder, or \"clean up\" the text in any way.",
#         "5. For parameters of type \"boolean\": output true or false (lowercase, unquoted).",
#         "6. Output ONLY a single JSON object with exactly two keys: \"name\" and \"parameters\".",
#         "   No prose, no explanation, no markdown, no trailing text.",
#         "",
#         "Available functions:",
#     ]
#     for f in functions:
#         params = ', '.join(
#             f"{name}: {info['type']}" for name, info in f['parameters'].items())
#         lines.append(f"   - {f['name']}({params}): {f['description']}")
#     lines.append('\nOutput ONLY valid JSON: {"name":"<fn>","args":"<args>"}')
#     return "\n".join(lines)


# if __name__ == "__main__":
#     parser = Parser()

#     # parsing args
#     p = parser.action()
#     f = FileLoader()
#     prompts = f.load_file_data(p.input, 'input')
#     functions_definition = f.load_file_data(
#         p.functions_definition, 'functions_definition')

#     try:
#         model = Small_LLM_Model()
#     except OSError as e:
#         print(e)

#     system_prompt = build_sytem_prompt(functions_definition)
#     vocabs_org = get_vocabs(model)
#     vocabs = {v: k for k, v in vocabs_org.items()}
#     output_data = list()
#     for prompt in prompts:
#         sytem_prompt_for_each_prompt = system_prompt + \
#             f"\nUser Prompt: {prompt}\nAnswer: "
#         tensor_2d_object = model.encode(sytem_prompt_for_each_prompt)
#         tokens = tensor_2d_object[0].tolist()
#         limit_index_fn_name = 0
#         limited_tokens = constrained_decoding(
#             functions_definition, model)
#         a = generate_token(vocabs, tokens, model, limited_tokens)
#         a = a.replace('\\', '\\\\')
#         a = a.replace("args", "parameters")
#         output = prompt | json.loads(a)
#         output_data.append(output)
#     try:
#         os.makedirs("data/output")
#     except FileExistsError as e:
#         pass
#     with open(p.output, 'w') as file:
#         json.dump(output_data, file, indent=4)
from Parser import Parser
from FileLoader import FileLoader
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import os
from itertools import zip_longest


def isValid(s: str) -> bool:
    if not s:
        return False
    new_s = str()
    for ch in s:
        if ch in ['{', '}']:
            new_s += ch

    stack = []
    mapping = {"}": "{"}

    for char in new_s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)

    return not stack


def set_limited_tokens_for_args(functions_definition, target_name):
    new_limited_tokens = []
    list_params_names = [
        param for f in functions_definition for param in f['parameters'].keys() if f['name'] == target_name]
    for token in limited_tokens:
        if token is None:
            for i, param in enumerate(list_params_names):
                new_limited_tokens += model.encode(
                    f"\"{param}\":")[0].tolist()
                new_limited_tokens.append(None)
        else:
            new_limited_tokens.append(token)
    return new_limited_tokens


def get_masked_logits(logits, next_expected_tokens) -> list[float] | None:

    if isinstance(next_expected_tokens, int):
        masked_logits = [score if index == next_expected_tokens else float(
            '-inf') for index, score in enumerate(logits)]
    elif isinstance(next_expected_tokens, list):
        masked_logits = [score if index in next_expected_tokens else float(
            '-inf') for index, score in enumerate(logits)]
    elif next_expected_tokens is None:
        masked_logits = logits
    else:
        masked_logits = None
    return masked_logits


def generate_token(vocabs, tokens, model, limited_tokens):
    i = 0
    global limit_index_fn_name
    function_name = str()
    flag = True
    answer = ""
    auto_generation = False
    # FIX (test 11): track whether we're currently inside a string VALUE,
    # so an internal `"` (e.g. Say "hello" to {name}) isn't mistaken
    # for the end of the string. We only treat a quote as closing
    # the string if it is NOT escaped (not preceded by a backslash).
    in_string_value = False
    while not isValid(answer):
        logits = model.get_logits_from_input_ids(tokens)
        n_logits = get_masked_logits(
            logits, limited_tokens[i] if not auto_generation and i < len(limited_tokens) else None)
        if n_logits is None:
            return

        if n_logits == logits and not auto_generation:
            auto_generation = True

        max_token = max(vocabs.keys(), key=lambda x: n_logits[x])
        if i >= 3 and flag:
            if function_name in [f['name'] for f in functions_definition]:
                i = limit_index_fn_name
                flag = False
                continue
            else:
                function_name += model.decode([max_token])

        elif not flag:
            limited_tokens = set_limited_tokens_for_args(
                functions_definition, function_name)

        decoded_piece = model.decode([max_token])
        answer += decoded_piece
        tokens.append(max_token)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(model.decode(tokens))
        if not auto_generation:
            i += 1

        # FIX (test 11): only end auto-generation on a quote token
        # if we're not inside an escaped sequence. We approximate
        # "escaped" by checking if the previous character emitted
        # was a backslash.
        if max_token in [1335, 2198] and auto_generation:
            prev_char = answer[-2] if len(answer) >= 2 else ""
            if not in_string_value:
                # this quote OPENS the string value
                in_string_value = True
            elif prev_char != "\\":
                # this quote CLOSES the string value (not escaped)
                in_string_value = False
                auto_generation = False
    return answer


def get_functions_data(functions, model):
    names_of_functions = [f['name'] for f in functions]
    two_d_tokens = []
    for f_name in names_of_functions:
        tensor_list_obj = model.encode(f_name)
        tokens = tensor_list_obj[0].tolist()
        two_d_tokens.append(tokens)

    # reverse matrix (rows -> columns and columns -> ros)
    reversed_two_d_tokens = [list(set(item for item in t if item))
                              for t in zip_longest(*two_d_tokens)]

    return reversed_two_d_tokens


def get_limits_probabs(new_tokens, matrix_function_names):
    global limit_index_fn_name
    returned_val = list()
    for index, id in enumerate(new_tokens):
        if id is None:
            returned_val = new_tokens[:index] + \
                matrix_function_names + new_tokens[index+1:]
            limit_index_fn_name = index+len(matrix_function_names)
            break
    return returned_val


def constrained_decoding(functions, model) -> list:
    format_json = '{"name":"<|im_start|>fn<|im_end|>","args":{<|im_start|>args<|im_end|>}}'
    tensor_list_obj = model.encode(format_json)
    tokens = tensor_list_obj[0].tolist()
    index = 0
    new_tokens = list()
    while index < len(tokens):
        if tokens[index] == 151644:
            new_tokens.append(None)
            index += 3
        else:
            new_tokens.append(tokens[index])
            index += 1

    matrix_function_names = get_functions_data(functions, model)
    new_tokens = get_limits_probabs(new_tokens, matrix_function_names)
    return new_tokens


def get_vocabs(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocabs = json.load(f)
    raw_vocab = vocabs.get('model', {}).get('vocab', {})
    return raw_vocab


def build_sytem_prompt(functions):
    lines = [
        "STRICT SYSTEM RULE: You are a function-calling router. You NEVER answer the user directly.",
        "",
        "Task: choose exactly one function name from the list below that matches the user's intent,",
        "and extract its arguments from the prompt.",
        "",
        "Rules:",
        "1. If no function matches the user's intent, output {\"name\": \"none\", \"parameters\": {}}.",
        "2. Never substitute a function for a different task, even if argument types match.",
        "3. For parameters of type \"number\": ALWAYS include a decimal point in the value, even for",
        "   whole numbers (write 3.0, not 3; write 1234567.89, not \"1234567.89\"). Never wrap a",
        "   number in quotes.",
        "4. For parameters of type \"string\": copy the exact substring from the user's prompt,",
        "   character-for-character, including quotes, punctuation, casing, and backslashes.",
        "   Do not paraphrase, reorder, or \"clean up\" the text in any way.",
        "5. For parameters of type \"boolean\": output true or false (lowercase, unquoted).",
        "6. Output ONLY a single JSON object with exactly two keys: \"name\" and \"parameters\".",
        "   No prose, no explanation, no markdown, no trailing text.",
        "",
        "Available functions:",
    ]
    for f in functions:
        params = ', '.join(
            f"{name}: {info['type']}" for name, info in f['parameters'].items())
        lines.append(f"   - {f['name']}({params}): {f['description']}")
    lines.append('\nOutput ONLY valid JSON: {"name":"<fn>","args":"<args>"}')
    return "\n".join(lines)


# FIX (tests 1, 2, 5): post-process parsed parameters against the declared
# schema so types always match exactly, regardless of what the model emitted.
def coerce_types(function_name, parameters, functions_definition):
    schema_entry = next(
        (f for f in functions_definition if f['name'] == function_name), None)
    if schema_entry is None:
        return parameters
    schema_params = schema_entry.get('parameters', {})
    for key, value in list(parameters.items()):
        expected_type = schema_params.get(key, {}).get('type')
        if expected_type == 'number':
            try:
                parameters[key] = float(value)
            except (TypeError, ValueError):
                pass
        elif expected_type == 'boolean' and isinstance(value, str):
            parameters[key] = value.strip().lower() == 'true'
        elif expected_type == 'integer':
            try:
                parameters[key] = int(float(value))
            except (TypeError, ValueError):
                pass
    return parameters


if __name__ == "__main__":
    parser = Parser()

    # parsing args
    p = parser.action()
    f = FileLoader()
    prompts = f.load_file_data(p.input, 'input')
    functions_definition = f.load_file_data(
        p.functions_definition, 'functions_definition')

    try:
        model = Small_LLM_Model()
    except OSError as e:
        print(e)

    system_prompt = build_sytem_prompt(functions_definition)
    vocabs_org = get_vocabs(model)
    vocabs = {v: k for k, v in vocabs_org.items()}
    output_data = list()
    for prompt in prompts:
        sytem_prompt_for_each_prompt = system_prompt + \
            f"\nUser Prompt: {prompt}\nAnswer: "
        tensor_2d_object = model.encode(sytem_prompt_for_each_prompt)
        tokens = tensor_2d_object[0].tolist()
        limit_index_fn_name = 0
        limited_tokens = constrained_decoding(
            functions_definition, model)
        a = generate_token(vocabs, tokens, model, limited_tokens)

        # FIX (test 9): removed the blind `a.replace('\\', '\\\\')`.
        # It double-escaped already-correct JSON and corrupted backslashes
        # (e.g. C:\Users\john -> C:\\Users\\john after parsing).
        # If raw generation is not valid JSON on its own, that must be
        # fixed at the grammar/decoding level, not patched here.

        parsed = json.loads(a)

        # FIX: rename key on the parsed dict, not via string replace,
        # so we never accidentally corrupt a value that happens to
        # contain the substring "args".
        if "args" in parsed:
            parsed["parameters"] = parsed.pop("args")

        parsed["parameters"] = coerce_types(
            parsed.get("name"), parsed.get("parameters", {}), functions_definition)

        output = prompt | parsed
        output_data.append(output)
    try:
        os.makedirs("data/output")
    except FileExistsError:
        pass
    with open(p.output, 'w') as file:
        json.dump(output_data, file, indent=4)