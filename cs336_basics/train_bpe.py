import os
from pathlib import Path
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time


def chunking(
        input_path: str|os.PathLike,
        special_tokens: list[str]
):
    """
    将长文本chunk化，分别预处理
    """
    file_size_bytes = os.path.getsize(input_path)
    target_chunk_size_bytes = 512 * 1024 * 1024
    desired_num_chunks = file_size_bytes//target_chunk_size_bytes + 1
    split_special_tokens = "<|endoftext|>".encode("utf-8")

    pre_token_count = {}

    with open(input_path,"rb") as file:
        boundaries = find_chunk_boundaries(file,desired_num_chunks,split_special_tokens)
        for start, end in zip(boundaries[:-1],boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8",errors="ignore")
            corpus = preparation(chunk,special_tokens)
            pre_token_count = pre_tokenization(corpus,pre_token_count)

    return pre_token_count


def preparation(
    text: str,
    special_tokens: list[str]
):
    """
    读取语料，并按special_tokens划分
    """
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
        pre_token_count: dict
):
    """
    对划分后的每部分做pre-tokenization，转换str得到dict[tuple[bytes,...],int]
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for sentence in corpus:
        for match in re.finditer(PAT,sentence):
            token = match.group()
            token = str_to_tuple_bytes(token)
            pre_token_count[token] = pre_token_count.get(token,0) + 1

    return pre_token_count


# def pair_bytes(
#         pre_token_count: dict
# ):
#     """
#     利用dict[tuple[bytes,...],int]计算dict[tuple(bytes,bytes),int]
#     """
#     bytes_pair_count = {}

#     for pre_token, count in pre_token_count.items():
#         if len(pre_token) < 2:
#             continue

#         for i in range(0,len(pre_token)-1):
#             pair = (pre_token[i],pre_token[i+1])
#             bytes_pair_count[pair] = bytes_pair_count.get(pair,0) + count

#     return bytes_pair_count


# def update_pretoken(
#         pre_token_count: dict,
#         best_pair: tuple,
#         new_token: bytes
# ):
#     """
#     将频数最高的pair合并，修改pre_token_count
#     """
#     new_count = {}
#     for token, count in pre_token_count.items():
#         new_pretoken = []
#         i = 0

#         while i < len(token):
#             if i < len(token) - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
#                 new_pretoken.append(new_token)
#                 i += 2
#             else:
#                 new_pretoken.append(token[i])
#                 i += 1

#         new_pretoken = tuple(new_pretoken)
#         new_count[new_pretoken] = new_count.get(new_pretoken,0) + count

#     return new_count


def merge(
        input_path: str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
):
    """
    Version1: 先建立初始词表，包含0-255和special_tokens，然后进行多轮merge，重算bytes_pair_count，选出频数最大的pair，更新vocab和merges，更新pre_token_count
    Version2: 建立初始词表，算一次bytes_pair_count，建立pretoken_id到当前状态和count的索引，建立pair到包含这个pair到pretoken_id到索引，在merge时只进行增量更新以节省运行时间
    """
    vocab = {}
    for i in range(256):
        vocab[i] = i.to_bytes()
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    if len(vocab) > vocab_size:
        raise ValueError("vocab_size is too small!")

    start = time.perf_counter()

    merges = []
    pre_token_count = chunking(input_path,special_tokens)
    pt_stat: list[list[bytes]] = []
    pt_count: list[int] = []
    bytes_pair_count: dict[tuple[bytes,bytes],int] = {}
    pair_in_pretokens: dict[tuple[bytes,bytes],set[int]] = {}
    pre_token_id = -1
    for pre_token, count in pre_token_count.items():
        pt_stat.append(list(pre_token))
        pt_count.append(count)
        pre_token_id += 1
        if len(pre_token) < 2:
            continue

        for i in range(0,len(pre_token)-1):
            pair = (pre_token[i],pre_token[i+1])
            bytes_pair_count[pair] = bytes_pair_count.get(pair,0) + count
            pair_in_pretokens.setdefault(pair,set()).add(pre_token_id)

    pretoken_time = int((time.perf_counter() - start) / 60)
    print(f"After {pretoken_time} minutes...")
    print("Starting merging...")

    while len(vocab) < vocab_size:
        if not bytes_pair_count:
            break
        
        best_pair = max(bytes_pair_count,key=lambda x: (bytes_pair_count[x],x))
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        todo_pretoken = set(pair_in_pretokens[best_pair])
        for id in todo_pretoken:
            pretoken = pt_stat[id]
            weight = pt_count[id]

            old_pair: dict[tuple[bytes,bytes],int] = {}
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i],pretoken[i+1])
                old_pair[pair] = old_pair.get(pair,0) + 1
            
            new_pretoken: list[bytes] = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and pretoken[i] == best_pair[0] and pretoken[i + 1] == best_pair[1]:
                    new_pretoken.append(new_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            new_pair: dict[tuple[bytes,bytes],int] = {}
            for i in range(len(new_pretoken) - 1):
                pair = (new_pretoken[i],new_pretoken[i+1])
                new_pair[pair] = new_pair.get(pair,0) + 1

            changed_pair = old_pair.keys() | new_pair.keys()
            for pair in changed_pair:
                delta = (new_pair.get(pair,0) - old_pair.get(pair,0)) * weight
                bytes_pair_count[pair] = bytes_pair_count.get(pair,0) + delta
                if bytes_pair_count[pair] == 0:
                    del bytes_pair_count[pair]
                elif bytes_pair_count[pair] < 0:
                    raise RuntimeError(f"Negative pair count: {pair}")

            old_pair_set = set(old_pair)
            new_pair_set = set(new_pair)
            removed_pair = old_pair_set - new_pair_set
            for pair in removed_pair:
                pair_in_pretokens[pair].discard(id)
                if not pair_in_pretokens[pair]:
                    del pair_in_pretokens[pair]
            added_pair = new_pair_set - old_pair_set
            for pair in added_pair:
                pair_in_pretokens.setdefault(pair,set()).add(id)

            pt_stat[id] = new_pretoken
            

        if len(vocab) % 100 == 0:
            now_time = int((time.perf_counter() - start) / 60)
            print(f"After {now_time} minutes...")
            print(f"Vocab length: {len(vocab)}, merges length: {len(merges)}")

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
