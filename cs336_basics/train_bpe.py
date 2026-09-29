import os
from pathlib import Path
import regex as re



def preparation(
    input_path: str|os.PathLike,
    special_tokens: list[str]
):
    """
    读取语料，并按special_tokens划分
    """

    text = Path(input_path).read_text(encoding="utf-8")

    if not special_tokens:
        return [text]

    re_special_tokens = sorted(special_tokens,key=len,reverse=True)
    re_special_tokens = [re.escape(token) for token in re_special_tokens]
    corpus = re.split("|".join(re_special_tokens),text)
    return corpus


def str_to_tuple_bytes(
        word: str,
):
    """
    将str转换为对应的tuple(bytes,...)
    """
    word_utf8 = word.encode(encoding="utf-8")
    by = tuple(b.to_bytes() for b in word_utf8)
    return by


def pre_tokenization(
        corpus: list[str],
):
    """
    对划分后的每部分做pre-tokenization，转换str得到dict[tuple[bytes,...],int]
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_count = {}
    for sentence in corpus:
        for match in re.finditer(PAT,sentence):
            token = match.group()
            token = str_to_tuple_bytes(token)
            pre_token_count[token] = pre_token_count.get(token,0) + 1

    return pre_token_count


def pair_bytes(
        pre_token_count: dict
):
    """
    利用dict[tuple[bytes,...],int]计算dict[tuple(bytes,bytes),int]
    """
    bytes_pair_count = {}

    for pre_token, count in pre_token_count.items():
        if len(pre_token) < 2:
            continue

        for i in range(0,len(pre_token)-1):
            pair = (pre_token[i],pre_token[i+1])
            bytes_pair_count[pair] = bytes_pair_count.get(pair,0) + count

    return bytes_pair_count


def update_pretoken(
        pre_token_count: dict,
        best_pair: tuple,
        new_token: bytes
):
    """
    将频数最高的pair合并，修改pre_token_count
    """
    new_count = {}
    for token, count in pre_token_count.items():
        new_pretoken = []
        i = 0

        while i < len(token):
            if i < len(token) - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                new_pretoken.append(new_token)
                i += 2
            else:
                new_pretoken.append(token[i])
                i += 1

        new_pretoken = tuple(new_pretoken)
        new_count[new_pretoken] = new_count.get(new_pretoken,0) + count

    return new_count


def merge(
        input_path: str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
):
    """
    先建立初始词表，包含0-255和special_tokens，然后进行多轮merge，重算bytes_pair_count，选出频数最大的pair，更新vocab和merges，更新pre_token_count
    """
    vocab = {}
    for i in range(256):
        vocab[i] = i.to_bytes()
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    if len(vocab) > vocab_size:
        raise ValueError("vocab_size is too small!")

    merges = []
    pre_token_count = pre_tokenization(preparation(input_path,special_tokens))


    while len(vocab) < vocab_size:
        bytes_pair_count = pair_bytes(pre_token_count)
        if not bytes_pair_count:
            break
        
        best_pair = max(bytes_pair_count,key=lambda x: (bytes_pair_count[x],x))
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        pre_token_count = update_pretoken(pre_token_count,best_pair,new_token)

    return (vocab,merges)

    

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.txt",
        help="Output file"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    special_tokens = ["<|endoftext|>"]
    result = merge(input_path,265,special_tokens)
    print(result)


if __name__ == "__main__":
    main()