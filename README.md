*This project has been created as part of the 42 curriculum by ychabane.*

# call me maybe — Introduction to function calling in LLMs

## Description

This project implements a **function calling tool** that translates natural language
prompts into structured, schema-compliant function calls using a small local LLM
(`Qwen/Qwen3-0.6B`, 0.6B parameters).

Given a prompt like *"What is the sum of 40 and 2?"*, the goal is not to have the model
answer `42` directly, but to have it produce a structured call:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

The core challenge is that small language models are unreliable at producing valid
JSON on their own (often below 30% success rate when just prompted). This project
solves that using **constrained decoding**: at every generation step, the model's
logits are masked so that only tokens compatible with valid JSON structure — and the
correct function-calling schema — can ever be selected. This guarantees 100% parseable,
schema-compliant output regardless of the model's raw reliability.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package (copied alongside this repository's `src/` folder)

### Installation

```bash
make install
# equivalent to: uv sync
```

### Running the project

```bash
make run
```

This runs, by default:
```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

You can override any path:
```bash
make run INPUT=data/input/other_tests.json
```

or call the module directly:
```bash
uv run python -m src --functions_definition <path> --input <path> --output <path>
```

### Other Makefile targets

| Target | Purpose |
|---|---|
| `make install` | Install dependencies via `uv sync` |
| `make run` | Run the main program |
| `make debug` | Run the program under Python's `pdb` debugger |
| `make lint` | Run `flake8` and `mypy` with the mandatory flags |

## Algorithm Explanation

### Overview of the pipeline

1. **Prompt construction** — a strict system prompt is built listing all available
   functions, their parameter types, and explicit output-format rules (always emit a
   decimal point for numbers, never quote numeric values, copy string values verbatim,
   output exactly two top-level keys). See `build_sytem_prompt`.
2. **Tokenization** — the full prompt (system instructions + user prompt) is encoded
   into token IDs using the model's tokenizer.
3. **Step-by-step generation with logit masking** — instead of letting the model choose
   freely at each step, `generate_token` retrieves the raw logits
   (`model.get_logits_from_input_ids`) and passes them through `get_masked_logits`,
   which sets every logit outside an allowed set to `-inf`. This guarantees the next
   token can only ever be one of the structurally valid choices.
4. **Skeleton-first decoding** — a fixed JSON skeleton
   (`{"name":"<fn>","args":{<args>}}`) is pre-tokenized once via `constrained_decoding`.
   The literal parts of the skeleton (braces, quotes, colons, keys) are forced
   token-by-token; the two "gap" positions (the function name, and the argument list)
   are marked with `None` and expanded dynamically:
   - the function-name gap is filled with a **restricted candidate set** built from the
     token-level "trie" of all valid function names (`get_functions_data` +
     `get_limits_probabs`), so the model can only ever spell out a real function name.
   - once a function name is confirmed, `set_limited_tokens_for_args` rebuilds the
     remaining skeleton to require exactly that function's declared parameter keys, in
     order, before falling back to free generation for each value.
5. **Free generation is not truly "free"** — after a parameter key is forced, the model
   generates its value token-by-token with no hard grammar restriction on the value
   itself (`auto_generation = True`), but generation of a string value is bounded by
   quote-tracking logic that distinguishes an opening quote, an internal escaped quote,
   and the true closing quote, so values containing embedded `"` characters are not
   mistaken for the end of the field.

### Why this satisfies "constrained decoding," not just prompting

The subject explicitly warns that prompting alone cannot reach 99%+ reliability with a
0.6B model. This project never *asks* the model to produce valid JSON — it makes invalid
JSON structurally unreachable by zeroing out the logits of every disallowed token before
selection happens, for the skeleton and the function name/parameter keys. The one
deliberate exception is argument *values*, where full vocabulary freedom is kept (so the
model can express arbitrary numbers and strings) but is safety-netted by post-generation
validation and type coercion rather than a rigid character grammar.

## Design Decisions

- **Skeleton pre-tokenization once per run, not per prompt** — the JSON skeleton and the
  set of valid function-name tokens never change across prompts (only the user's
  question does), so `constrained_decoding` is called once, before the prompt loop,
  instead of being recomputed for every prompt.
- **`lru_cache` on pure string→token helpers** — encoding a function name or a
  `"param":` key literal always produces the same token sequence for the same input
  string, with no dependency on external state. These calls
  (`encode_function_name`, `encode_param_key`, `decode_single_token`) are cached with
  `functools.lru_cache`, so repeated parameter names (`a`, `b`, `name`, ...) across
  different functions and prompts are tokenized only once per run. Functions taking
  unhashable arguments (`list`, `dict`, the model object itself) are deliberately **not**
  cached — caching those would either crash (`TypeError: unhashable type`) or provide no
  benefit since the input is different on every call.
- **Type coercion after parsing, not a full numeric grammar** — rather than building a
  character-level grammar that forces digit/`.`/`-` tokens only for `number` fields
  (a larger undertaking), the project applies schema-based coercion after JSON parsing.
  This is a pragmatic trade-off: simpler to implement and verify, at the cost of not
  being "pure" constrained decoding for values. This trade-off is documented here rather
  than hidden.
- **Pydantic for schema validation** — `Prompt` and `FunctionCall` models validate the
  input files' shape at load time (Chapter IV.3.1 requirement), and results are
  validated before being written to output, catching malformed or empty values early
  with clear errors instead of writing bad data.
- **Per-prompt failure isolation** — a single prompt failing to generate valid JSON does
  not abort the entire run; see Advanced Error Recovery below.

## Performance Analysis

- **Validity**: constrained decoding guarantees 100% of outputs are syntactically valid
  JSON (verified by `isValid` before ever attempting `json.loads`), satisfying the
  hard requirement in Chapter V.1.
- **Accuracy**: on the provided sample tests, function selection is constrained to only
  real function names, so misrouting to a nonexistent function is structurally
  impossible; argument-level accuracy depends on the free-generation portion for values
  and is improved by post-parse type coercion.
- **Speed**: generation is sequential, one token at a time, with a full forward pass per
  token (`get_logits_from_input_ids`) — the dominant cost is model inference, not the
  masking logic itself, which is O(vocab size) per step. Caching described above removes
  redundant tokenizer calls but does not reduce the number of model forward passes.
- **Reliability**: malformed or missing input files, missing required parameters, and
  generation timeouts are all handled without crashing the process (see below).

## Challenges Faced

- **Numbers losing their decimal point** (`3` generated instead of `3.0`, or a number
  wrapped in quotes) — the free-generation portion for values has no type-aware
  constraint, so the model would occasionally emit an integer-looking token sequence or
  quote a number. Solved with post-parse type coercion against the declared schema.
- **String values with embedded quotes being truncated** (e.g. `Say "hello" to {name}`
  losing its quotes and being reordered) — the initial quote-termination logic treated
  the *first* quote character encountered mid-string as the end of the value. Fixed by
  tracking string-open/close state explicitly and only closing on an unescaped quote
  once already inside a string value.
- **Corrupted backslash-containing paths** (e.g. Windows paths like
  `C:\Users\john\config.ini`) — an unconditional `str.replace('\\', '\\\\')` on the
  already-valid generated JSON text was double-escaping backslashes before parsing.
  Removed entirely once the underlying generation was confirmed to already emit
  correctly escaped JSON.
- **`lru_cache` crashing with `TypeError: unhashable type: 'list'`** — initially applied
  directly to a function taking `functions_definition` (a list) and `model` as
  arguments. Fixed by only caching the small pure helper functions that take hashable
  arguments (plain strings/ints), and instead hoisting the expensive-but-repeatable call
  out of the per-prompt loop rather than trying to cache it.

## Testing Strategy

- **Unit tests** (`tests/`, run via `make test`) cover:
  - `isValid` brace-balance detection on valid, truncated, and empty input.
  - Type coercion (`coerce_types`) for numbers passed as `int`, as numeric strings, and
    booleans passed as string literals.
  - Tokenizer/cache helper round-trips where applicable.
- **Manual validation** against the provided `function_calling_tests.json` /
  `functions_definition.json` examples, checking the produced
  `function_calling_results.json` is valid JSON and that every entry's parameter types
  match the function's declared schema.
- **Edge cases exercised**: empty strings, large numbers, prompts with no matching
  function (`"name": "none"`), strings containing quotes and backslashes, and functions
  with multiple parameters.

## Advanced Error Recovery

Failures are isolated to the smallest possible scope rather than aborting the whole run:

- **Input file loading** — missing files, invalid JSON, and wrong encodings are caught
  individually with a clear, specific message before the program exits cleanly, instead
  of an unhandled traceback.
- **Per-prompt isolation** — if generation or JSON parsing fails for one prompt, that
  failure is logged and a fallback `{"name": "none", "parameters": {}}` result is
  recorded for that prompt so it still appears in the output file; the remaining prompts
  continue processing normally.
- **State-safe retries** — when a generation attempt is retried, it operates on a fresh
  copy of the token list and constraint list rather than reusing state mutated by the
  failed attempt, avoiding a retry that repeats the same failure from corrupted state.

## Example Usage

Given `data/input/functions_definition.json`:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

and `data/input/function_calling_tests.json`:
```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

Running:
```bash
make run
```

produces `data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Resources

- [GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=5149s) create a simple tokenizer
- [Build simple Gpt](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=963s) learn how gpt works 

- [Byte Pair Encoding](https://www.youtube.com/watch?v=hL4ZnAWSyuU&t=32s)


