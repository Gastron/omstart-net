#!/usr/bin/env python3

def read_reverse_vocab(filepath):
    with open(filepath, "r") as fi:
        return {i: word for word, i in (line.strip().split() for line in fi)}

def decode(line, vocab):
    return [vocab[char] for char in line.strip().split()]

def textify_plus(tokens):
    spaced = " ".join(tokens)
    return spaced.replace("+ +", "")

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser("""Convert a tokenized and integerized text back to text.
            """)
    parser.add_argument("vocab", help = "vocabulary file, mapping token to integer")
    parser.add_argument("input", nargs = "+", help = "file(s) to process or - for stdin")
    parser.add_argument("--keep-tags", help = "Keep sentence beginning and ending tags",
            const = True,
            action = "store_const",
            default = False)
    parser.add_argument("--basic-tokenisation", 
            help = "Set this flag if text was tokenized without the adjoining plus symbols",
            const = True,
            action = "store_const",
            default = False)
    args = parser.parse_args()
    vocab = read_reverse_vocab(args.vocab)
    for line in fileinput.input(args.input):
        tokens = decode(line, vocab)
        without_tags = tokens[1:-1]
        if args.basic_tokenisation:
            textified = "".join(without_tags)
        else:
            textified = textify_plus(without_tags)
        if args.keep_tags:
            print(tokens[0] + textified + tokens[-1])
        else:
            print(textified)

