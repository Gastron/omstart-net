#!/usr/bin/env python3

def read_reverse_vocab(filepath, space_symbol):
    with open(filepath, "r") as fi:
        return {i: word if word != space_symbol else " " 
                for word, i in (line.strip().split() for line in fi)}

def decode(line, vocab, tags_in_text=False):
    if tags_in_text:
        tokens = line.strip().split()
        return [tokens[0]] + [vocab[char] for char in tokens[1:-1]] + [tokens[-1]]
    else:
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
            action = "store_true")
    parser.add_argument("--tags-in-text", 
            help = "Start symbols were left as they are. e.g. for SRILM",
            action = "store_true")
    parser.add_argument("--no-tags",
            help = "Text has no tags",
            action = "store_true")
    parser.add_argument("--space-symbol",
            default = "<space>",
            help = "symbol for space, this is printed as the space character")
    args = parser.parse_args()
    vocab = read_reverse_vocab(args.vocab, args.space_symbol)
    for line in fileinput.input(args.input):
        tokens = decode(line, vocab, tags_in_text=args.tags_in_text)
        without_tags = tokens[1:-1] if not args.no_tags else tokens
        if args.basic_tokenisation:
            textified = "".join(without_tags)
        else:
            textified = textify_plus(without_tags)
        if args.keep_tags:
            print(tokens[0] + textified + tokens[-1])
        else:
            print(textified)

