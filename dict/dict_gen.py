#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@version:
@software:PyCharm
@file:dict_gen.py
@time:2020/10/27 16:29
"""

def event_dict_gen():
    files = ["../data-ACE/example_new.train", "../data-ACE/example_new.dev", "../data-ACE/example_new.test"]
    vocab_set = set()
    vocab_set_ner_1 = set()
    vocab_set_ner_2 = set()
    for filei in files:
        with open(filei, encoding="utf-8", mode="r") as f:
            for line in f:
                line = line.strip().split(" ")
                if len(line) == 5:
                    vocab_set.add(line[-1])
                    vocab_set_ner_1.add(line[-3])
                    vocab_set_ner_2.add(line[-2])
    vocab_list = list(vocab_set)
    vocab_list = [x for x in vocab_list if x == "O" or x.startswith("B-")]
    vocab_list += ["I-" + x[2:] for x in vocab_list if x.startswith("B-")]
    vocab_list = sorted(vocab_list, key=lambda x: x, reverse=True)
    with open("event_types.txt", encoding="utf-8", mode="w") as fw:
        for line in vocab_list:
            fw.write(line + "\n")

    vocab_list_ner_1 = list(vocab_set_ner_1)
    vocab_list_ner_1 = sorted(vocab_list_ner_1, key=lambda x: x, reverse=True)
    vocab_set_ner_2 = list(vocab_set_ner_2)
    vocab_set_ner_2 = sorted(vocab_set_ner_2, key=lambda x: x, reverse=True)

    with open("ner_1.txt", encoding="utf-8", mode="w") as fw:
        for line in vocab_list_ner_1:
            fw.write(line + "\n")

    with open("ner_2.txt", encoding="utf-8", mode="w") as fw:
        for line in vocab_set_ner_2:
            fw.write(line + "\n")


if __name__ == "__main__":
    event_dict_gen()




