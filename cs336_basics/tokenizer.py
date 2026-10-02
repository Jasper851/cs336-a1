from collections.abc import Iterable, Iterator
import pickle
from pathlib import Path
import regex as re
import cs336_basics.train_bpe as train_bpe

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode(encoding="utf-8")
                if token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token_bytes

            self.special_tokens = sorted(self.special_tokens,key=len,reverse=True)
        else:
            self.special_tokens = []

        self.reversed_vocab = {v : k for k,v in self.vocab.items()}
        self.merge_ranks = {pair : rank for rank, pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as file:
            vocab: dict[int, bytes] = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges: list[tuple[bytes, bytes]] = pickle.load(file)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = f"({"|".join(re.escape(token) for token in self.special_tokens)})"
            corpus = re.split(pattern,text)
        else:
            corpus = [text]

        encode_ids: list[int] = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in corpus:
            if not part:
                continue

            if part in self.special_tokens:
                encode_ids.append(self.reversed_vocab[part.encode(encoding="utf-8")])
            else:
                for match in re.finditer(PAT,part):
                    token = match.group()
                    token = list(train_bpe.str_to_tuple_bytes(token))

                    while len(token) >= 2:
                        current_pairs = set(zip(token,token[1:]))
                        applicable_pairs = [pair for pair in current_pairs if pair in self.merge_ranks.keys()]

                        if not applicable_pairs:
                            break

                        best_pair = min(applicable_pairs,key=lambda pair: self.merge_ranks[pair])
                        merged_tokens: list[bytes] = []
                        i = 0
                        while i < len(token):
                            if i + 1 < len(token) and (token[i],token[i+1]) == best_pair:
                                merged_tokens.append(token[i]+token[i+1])
                                i += 2
                            else:
                                merged_tokens.append(token[i])
                                i += 1

                        token = merged_tokens

                    encode_ids += [self.reversed_vocab[t] for t in token]

        return encode_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:
        text_bytes = bytes()
        for i in ids:
            text_bytes += self.vocab[i]

        text_str = text_bytes.decode(encoding="utf-8",errors="replace")
        return text_str

