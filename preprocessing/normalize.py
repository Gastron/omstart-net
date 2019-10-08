#!/usr/bin/env python3
import re
import string
import unicodedata
#all Finnish letters, space and -

def normalize(text, keep_case=False, keep_nums=False, extra_chars=""):
    CHARSET = set(string.ascii_lowercase) | set("åäö- ") | set(extra_chars)
    if keep_case:
        CHARSET |= set(string.ascii_uppercase) | set("ÅÄÖ")
    else:
        text = text.lower()
    if keep_nums:
        CHARSET |= set(string.digits)
    #remove accents, etc.:
    text = "".join(char if char in CHARSET else unicodedata.normalize("NFKD", char) for char in text)
    #filter all not in charset finally:
    text = "".join(char for char in text if char in CHARSET)
    #all whitespace to one space, and strip leading/trailing whitespace:
    text = re.sub(r"\s+"," ",text) 
    text = text.strip()
    return text

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser("Normalize chars so as to keep rare chars out of vocab")
    parser.add_argument("--keep-case", help = "By default, lowercases everything.",
            action = "store_true")
    parser.add_argument("--keep-nums", help = "By default, delete all number tokens",
            action = "store_true")
    parser.add_argument("--extra-chars", type=str, help = "A string of chars to also include",
            default = "")
    parser.add_argument("input", nargs="+", help = "File(s) to process or - for stdin")
    args = parser.parse_args()
    for line in fileinput.input(args.input):
        print(normalize(line, args.keep_case, args.keep_nums,
            args.extra_chars))
