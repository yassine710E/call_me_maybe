import json
from llm_sdk import Small_LLM_Model
class Tokenizer:
    def __init__(self, text: str, vocabs: dict):
        self.text = text.replace(' ', 'Ġ')
        self.vocabs = vocabs
        self.tokens = list()

    def encode(self):
        i = 0
        l_chunks = []
        n = len(self.text)
        max_token_len = max((len(k) for k in self.vocabs), default=1)

        while i < n:
            match = None
            # try the longest possible substring first, shrink until one matches
            for length in range(min(max_token_len, n - i), 0, -1):
                candidate = self.text[i:i + length]
                if candidate in self.vocabs:
                    match = candidate
                    break
            if match is None:
                # no vocab entry at all for this character -- fallback,
                # consume one char to avoid an infinite loop
                match = self.text[i]
            l_chunks.append(match)
            i += len(match)

        self.tokens = [self.vocabs[ch] for ch in l_chunks]
        return self.tokens

    def decode(self):
        re_vocabs = {v: k for k, v in self.vocabs.items()}
        s = "".join(re_vocabs[token] for token in self.tokens)
        return s.replace('Ġ', ' ')    
def get_vocabs(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocabs = json.load(f)
    raw_vocab = vocabs.get('model', {}).get('vocab', {})
    return raw_vocab

model = Small_LLM_Model()
vocabs = get_vocabs(model)


t = Tokenizer("What is the sum of 2 and 3?", vocabs)
mine = t.encode()
real = model.encode("What is the sum of 2 and 3?")[0].tolist()
print(mine)
print(real)
print(mine == real)