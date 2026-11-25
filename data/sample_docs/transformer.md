# The Transformer Language Model

## Tokenization
Text is converted to integers by a byte pair encoding (BPE) tokenizer. Training starts from the 256 possible bytes and repeatedly merges the most frequent adjacent pair into a new token until the vocabulary size is reached, for example 32768 tokens. A regular expression first splits text into words, numbers and punctuation so merges never cross those boundaries. Special tokens such as the beginning-of-sequence token and user/assistant start and end markers delimit chat turns.

## Attention
Self-attention lets every token look at earlier tokens. Each token is projected to a query, a key and a value. Attention weights are the softmax of query-key dot products scaled by the square root of the head dimension, and the output is the weighted sum of values. A causal mask stops tokens from attending to the future. Multi-head attention runs several such heads in parallel. Group-query attention shares each key/value head among several query heads, which shrinks the KV cache. Sliding-window attention restricts some layers to a recent window of tokens.

## Position information
Rotary position embeddings rotate pairs of query and key dimensions by an angle proportional to the token position, so the dot product depends on the relative distance between tokens. QK normalisation applies RMSNorm to queries and keys to keep attention logits stable.

## Blocks and generation
A transformer block applies attention and then an MLP, each with a residual connection and pre-normalisation. The final hidden state is projected to vocabulary logits. Generation samples one token at a time: temperature scales the logits, top-k keeps only the k most likely tokens, and greedy decoding takes the arg max. Generation stops at an end token or after a maximum number of tokens.
