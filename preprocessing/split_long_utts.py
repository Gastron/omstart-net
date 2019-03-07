#!/usr/bin/env python3

def split_long_utt(text, maxchars):
    parts = []
    splits_to_be_made = len(text) // maxchars
    while splits_to_be_made > len(parts):
        parts.append(text[len(parts)*maxchars:(len(parts)+1)*maxchars])
    parts.append(text[len(parts)*maxchars:])
    return parts

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser("Split long training samples to multiple shorter ones")
    parser.add_argument("--maxchars", type=int, default=30)
    parser.add_argument("input", nargs = "+", help = "file(s) to process or - for stdin")
    args = parser.parse_args()
    for line in fileinput.input(args.input):
        line = line.strip()
        utts = split_long_utt(line, maxchars = args.maxchars)
        for utt in utts:
            print(utt)
