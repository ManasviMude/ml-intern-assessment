# Evaluation — TrigramModel

Approach:
- Implemented a trigram language model using raw unigram/bigram/trigram counts.
- Tokenization: basic whitespace tokenizer with punctuation stripped from token edges.
- Sentences are split on `. ? !` and padded with two `<s>` tokens at start and a `</s>` token at end.
- Model accepts raw string input (fit on text) or an iterable of token lists.

Probabilities:
- MLE estimate: P(w3 | w1, w2) = C(w1,w2,w3) / C(w1,w2).
- Optional add-k smoothing available in trigram_prob (used for evaluation if needed).

Complexity:
- Training time: O(N) where N = number of tokens. Memory: proportional to number of observed n-grams.

Limitations & future improvements:
- No backoff/interpolation or Kneser-Ney smoothing — unseen trigrams get zero prob (address with smoothing/backoff).
- Could replace tokenizer with subword/BPE or use a neural LM (RNN/Transformer) for generation tasks and better generalization.
