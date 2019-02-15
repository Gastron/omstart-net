#!/usr/bin/env python3

def read_vocab(filepath):
    with open(filepath, "r") as fi:
        return {word: i for word, i in (line.strip().split() for line in fi)}

def apply_vocab(tokens, vocab, unk="<unk>"):
    return [vocab[token] if token in vocab else vocab[unk] for token in tokens]

def tokenize_basic(line):
    return list(line.strip())

def tokenize_plus(line):
    #Each char has a plus on the sides which do not touch a space.
    #ie. H+ +e+ +l+ +l+ +o w+ +o+ +r+ +l+ +d
    #Then you can simply: .replace("+ +", "") to obtain Hello world
    #Note! We don't want pluses to touch sentence start and end tokens.
    words = line.strip().split()
    words_with_pluses = ["+ +".join(word) for word in words]
    chars = " ".join(words_with_pluses)
    return chars.split()

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser("""Convert a text to character level tokens given a dictionary.
            By default, will use adjoining plus sumbols.
            ie. H+ +e+ +l+ +l+ +o w+ +o+ +r+ +l+ +d
            Then you can simply: sed "s/\+ \+//g" to obtain Hello world
            Note! We don't want pluses to touch sentence start and end tokens.
            """)
    parser.add_argument("vocab", help = "vocabulary file, mapping token to integer")
    parser.add_argument("input", nargs = "+", help = "file(s) to process or - for stdin")
    parser.add_argument("--start-symbol", default = "<s>", help = "sentence start symbol in vocab")
    parser.add_argument("--end-symbol", default = "</s>", help = "sentence end symbol in vocab")
    parser.add_argument("--unk", default = "<unk>", 
            help = "token for character not in the vocabulary")
    parser.add_argument("--basic-tokenisation", help = "Tokenize without the adjoining plus symbols",
            dest = "tokenizer", 
            action = "store_const",
            const = tokenize_basic, 
            default = tokenize_plus)
    parser.add_argument("--leave-tags", 
            help = "Leave sentence start symbols as they are. e.g. for SRILM",
            action = "store_true")
    args = parser.parse_args()
    vocab = read_vocab(args.vocab)
    for line in fileinput.input(args.input):
        tokens = args.tokenizer(line) 
        mapped_sentence = apply_vocab(tokens, vocab, args.unk)
        if args.leave_tags:
            with_tags = [args.start_symbol] + mapped_sentence + [args.end_symbol]
        else:
            with_tags = [vocab[args.start_symbol]] + mapped_sentence + [vocab[args.end_symbol]]
        print(" ".join(with_tags))
