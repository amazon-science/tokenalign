import transformers
from transformers import AutoTokenizer

tokenizer_name = "bigcode/starcoder"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

print(tokenizer.__dict__)

tokens = tokenizer.tokenize("def get_star():\n  return 'star'")
print(tokens)