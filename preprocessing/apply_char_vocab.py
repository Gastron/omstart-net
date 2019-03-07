#!/usr/bin/env python3

def read_vocab(filepath):
    with open(filepath, "r") as fi:
        return {word: i for word, i in (line.split() for line in fi)} 
def apply_vocab(tokens, vocab, unk="<unk>"):
    return [vocab[token] if token in vocab else vocab[unk] for token in tokens]

def tokenize_basic(line, space_symbol):
    chars = line.strip()
    return [char if char != " " else space_symbol for char in chars]

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
    parser.add_argument("--space-symbol", default = "<space>", help = "symbol for space")
    parser.add_argument("--unk", default = "<unk>", 
            help = "token for character not in the vocabulary")
    parser.add_argument("--basic-tokenisation", help = "Tokenize without the adjoining plus symbols",
            action = "store_true")
    parser.add_argument("--leave-tags", 
            help = "Leave sentence start symbols as they are. e.g. for SRILM",
            action = "store_true")
    args = parser.parse_args()
    if args.basic_tokenisation:
        tokenizer = lambda x: tokenize_basic(x, args.space_symbol)
    else:
        tokenize = tokenize_plus

    vocab = read_vocab(args.vocab)
    for line in fileinput.input(args.input):
        tokens = tokenizer(line) 
        mapped_sentence = apply_vocab(tokens, vocab, args.unk)
        if args.leave_tags:
            with_tags = [args.start_symbol] + mapped_sentence + [args.end_symbol]
        else:
            with_tags = [vocab[args.start_symbol]] + mapped_sentence + [vocab[args.end_symbol]]
        print(" ".join(with_tags))
