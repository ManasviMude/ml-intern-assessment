# src/ngram_model.py
from collections import Counter
import math
import re
from typing import List, Iterable, Tuple, Optional, Union


def _simple_tokenize(text: str) -> List[str]:
    """Very small tokenizer: split on whitespace and strip punctuation from token edges."""
    tokens = []
    for tok in text.strip().split():
        tok = tok.strip('.,;:"\'()[]{}')
        if tok:
            tokens.append(tok)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return a list of n-grams padded with <s> and </s>. Always treats tokens as a list."""
    if n < 1:
        raise ValueError("n must be >= 1")
    padded = ["<s>"] * (n - 1) + list(tokens) + ["</s>"]
    return [tuple(padded[i:i + n]) for i in range(len(padded) - n + 1)]


class TrigramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()

    def _text_to_sentences(self, text: str) -> List[List[str]]:
        """Split raw text into sentences (basic) and tokenize."""
        parts = re.split(r'[.?!]+', text)
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            toks = _simple_tokenize(part)
            if toks:
                sentences.append(toks)
        return sentences

    def fit(self, corpus: Optional[Union[str, Iterable[Union[str, Iterable[str]]]]] = None) -> None:
        """
        Train the model.
        Accepts:
          - a single raw string (text) -> splits into sentences and tokenizes
          - an iterable of token-lists (each element is a sentence tokens list)
          - an iterable of strings (each string is treated as one sentence)
        """
        token_lists: List[List[str]] = []

        if corpus is None:
            return

        # If corpus is a raw string -> split into sentence token lists
        if isinstance(corpus, str):
            token_lists = self._text_to_sentences(corpus)
        else:
            # it's an iterable - normalize each element into a token list
            for entry in corpus:
                if isinstance(entry, str):
                    token_lists.append(_simple_tokenize(entry))
                else:
                    token_lists.append(list(entry))

        # build counts
        for tokens in token_lists:
            for (w,) in get_ngrams(tokens, 1):
                self.unigrams[(w,)] += 1
            for bg in get_ngrams(tokens, 2):
                self.bigrams[bg] += 1
            for tg in get_ngrams(tokens, 3):
                self.trigrams[tg] += 1

        # build vocab (exclude padding tokens)
        self.vocab = {w for (w,) in self.unigrams if w not in ("<s>", "</s>")}

    def trigram_count(self, w1: str, w2: str, w3: str) -> int:
        return self.trigrams.get((w1, w2, w3), 0)

    def bigram_count(self, w1: str, w2: str) -> int:
        return self.bigrams.get((w1, w2), 0)

    def trigram_prob(self, w1: str, w2: str, w3: str, add_k: float = 0.0) -> float:
        """Return P(w3 | w1,w2) using MLE with optional add-k smoothing."""
        vocab_size = max(1, len(self.vocab)) + 1
        numerator = self.trigram_count(w1, w2, w3) + add_k
        denominator = self.bigram_count(w1, w2) + add_k * vocab_size
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def sentence_logprob(self, tokens: List[str], add_k: float = 0.0) -> float:
        total = 0.0
        for (w1, w2, w3) in get_ngrams(tokens, 3):
            p = self.trigram_prob(w1, w2, w3, add_k)
            if p <= 0:
                return float("-inf")
            total += math.log(p)
        return total

    def top_k(self, w1: str, w2: str, k: int = 5):
        candidates = []
        for w in list(self.vocab) + ["</s>"]:
            p = self.trigram_prob(w1, w2, w)
            if p > 0:
                candidates.append((w, p))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def generate(self, max_len: int = 50) -> str:
        """
        Greedy deterministic generation with a fallback:
        - Start with <s>, <s>.
        - If no trigram candidates have positive probability, fall back to the most common unigram.
        """
        if not self.trigrams:
            return ""

        w1, w2 = "<s>", "<s>"
        out: List[str] = []
        for _ in range(max_len):
            best_word = None
            best_p = 0.0
            for w in list(self.vocab) + ["</s>"]:
                p = self.trigram_prob(w1, w2, w)
                if p > best_p:
                    best_p = p
                    best_word = w

            # FALLBACK: if no candidate found (best_word is None), pick most frequent unigram
            if best_word is None:
                # find most common unigram excluding padding tokens
                most_common = [w for (w,), _ in self.unigrams.most_common() if w not in ("<s>", "</s>")]
                if not most_common:
                    break
                best_word = most_common[0]

            if best_word == "</s>":
                break

            out.append(best_word)
            w1, w2 = w2, best_word

        return " ".join(out)
