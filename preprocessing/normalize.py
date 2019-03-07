#!/usr/bin/env python3
import re
import string
import unicodedata
#all Finnish letters, space and -
CHARSET = set(string.ascii_lowercase) | set("åäö- ")

def normalize(text):
    text = text.lower()
    #normalise:
    text = "".join(char if char in CHARSET else unicodedata.normalize("NFKD", char) for char in text)
    #filter all not in charset finally:
    text = "".join(char for char in text if char in CHARSET)
    #all whitespace to one space, and strip leading/trailing whitespace:
    text = re.sub(r"\s+"," ",text) 
    text = text.strip()
    return text

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(normalize(line))
