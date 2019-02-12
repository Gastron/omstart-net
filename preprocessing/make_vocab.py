#!/usr/bin/env python3
from apply_char_vocab import tokenize_basic, tokenize_plus
import sys

def write_vocab(out_descriptor, vocab):
    for char, mapped in sorted(vocab.items(), key=lambda x:x[1]):
        print(char, mapped, file=out_descriptor)

def build_map(char_set, special_chars):
    ordered = special_chars + sorted(char_set)
    return {char: index for index, char in enumerate(ordered)}

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser("Make a vocab file based on a corpus")
    parser.add_argument("input", nargs = "+", help = "file(s) to process or - for stdin")
    parser.add_argument("--output", default = "-", help = "output to file, default: stdout")
    parser.add_argument("--start-symbol", default = "<s>", help = "sentence start symbol in vocab")
    parser.add_argument("--end-symbol", default = "</s>", help = "sentence end symbol in vocab")
    parser.add_argument("--unk", default = "", help = "token for character not in the vocabulary, default: NOT INCLUDED")
    parser.add_argument("--basic-vocab", help = "Vocab without the adjoining plus symbols",
            dest = "tokenizer", 
            action = "store_const",
            const = tokenize_basic, 
            default = tokenize_plus)
    args = parser.parse_args()
    char_set = set()
    for line in fileinput.input(args.input):
        char_set.update(args.tokenizer(line))

    special_chars = [args.end_symbol, args.start_symbol]
    if args.unk:
        special_chars.append(args.unk)
    vocab = build_map(char_set, special_chars)
    if args.output != "-":
        with open(args.output, "w") as fo:
            write_vocab(fo, vocab)
    else:
        write_vocab(sys.stdout, vocab)
