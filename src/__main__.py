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


def generate_token(vocabs, tokens, model,  limited_tokens):
    i = 0
    global limit_index_fn_name
    function_name = str()
    flag = True
    answer = ""
    auto_generation = False
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
        answer += model.decode([max_token])
        tokens.append(max_token)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(model.decode(tokens))
        # if not auto_generation:
        i += 1
        if max_token in [1335, 2198] and auto_generation:
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
        if id == None:
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
        'STRICT SYTEM RULE: Use ONLY a matching function from the list below.',
        "if NO function matches the user's intent (even if types match),set name: \"none\".",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
    ]
    for f in functions:
        params = ', '.join(
            f"{name}: {info['type']}" for name, info in f['parameters'].items())
        lines.append(f"   - {f['name']}({params}): {f['description']}")
    lines.append('\nOutput ONLY valid JSON: {"name":"<fn>","args":"<args>"}')
    return "\n".join(lines)


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
    for prompt in prompts:
        sytem_prompt_for_each_prompt = system_prompt + \
            f"\nUser Prompt: {prompt}\nAnswer: "
        tensor_2d_object = model.encode(sytem_prompt_for_each_prompt)
        tokens = tensor_2d_object[0].tolist()
        limit_index_fn_name = 0
        limited_tokens = constrained_decoding(
            functions_definition, model)
        a = generate_token(vocabs, tokens, model, limited_tokens)
        # print(prompt)
        break
        # print(a)
