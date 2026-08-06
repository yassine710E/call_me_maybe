from Parser import Parser
from FileLoader import FileLoader
from Parser import Prompt, FunctionCall
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import regex
from itertools import zip_longest


def constrained_decoding(functions: list[dict], logits: list[float], vocabs: dict[str:int]):
    # stage 1 : guide of function name
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    compiled_regex = regex.compile(pattern)
    chunks_format = compiled_regex.findall('{"name":"<fn>","args":"<args>"}')
    function_chunks = []
    args_chunks = []
    for f in functions:
        chunks = compiled_regex.findall(f['name'])
        args = [ch for name_arg in f['parameters'].keys()
                for ch in compiled_regex.findall(name_arg)]
        args_chunks.append(args)
        function_chunks.append(chunks)
    # this is parameters
    transposed_tuples = list(zip_longest(*args_chunks, fillvalue=None))
    output = list(map(list, transposed_tuples))
    limit_args = [set([s for s in item if s is not None])
                  for item in output]
    # this is for function
    transposed_tuples = list(zip_longest(*function_chunks, fillvalue=None))
    output = list(map(list, transposed_tuples))
    limit_functions = [set([s for s in item if s is not None])
                       for item in output]
    result = []
    flag_arg = True
    for chunk in chunks_format:
        if "<" in chunk:
            result.append(chunk.replace('<', ""))
        elif ">" in chunk:
            result.append(chunk.replace('>', ''))
        elif chunk == "fn":
            result += limit_functions
        elif chunk == "args" and flag_arg:
            result.append(chunk)
            flag_arg = False
        elif chunk == "args" and not flag_arg:
            result += limit_args
        else:
            result.append(chunk)
    print(result)


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
        stage_counter = 0

        for i in range(0, 400):
            logits = model.get_logits_from_input_ids(tokens)
            constrained_decoding(functions_definition, logits, vocabs_org)
            max_token = max(vocabs.keys(), key=lambda x: logits[x])
            tokens.append(max_token)
            print(model.decode(tokens))
            # print(model.decode([max_token]))
        break
