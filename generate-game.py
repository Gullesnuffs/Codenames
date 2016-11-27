#!/usr/bin/env python3
import sys
import random

if len(sys.argv) != 2:
    print("Usage: python3 generate-game.py wordlist.txt")
    print("The wordlist for the original game can be found at: https://boardgamegeek.com/article/21511664#21511664")
    print("Start by making it all-lowercase and remove the words with spaces (in vim: guG, :g/ /d).")
    exit(1)

with open(sys.argv[1]) as f:
    words = [x.strip() for x in f]
    chosen = random.sample(words, 25)
    start = random.choice("rb")
    affiliation = list(start + 8*"r" + 8*"b" + 7*"g" + "a")
    random.shuffle(affiliation)
    print(start)
    for (aff, word) in zip(affiliation, chosen):
        print("{} {}".format(aff, word))
    print("go")
