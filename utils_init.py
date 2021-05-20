#!/usr/bin/env python
#-*- coding: utf-8 -*-


import random
import numpy as np


def load_vocab(filename):
    vocab = []
    with open(filename,encoding='utf-8',mode='r') as f:
        for line in f:
            vocab.append(line.strip())
    vocab_to_id = {u:i for i,u in enumerate(vocab)}
    return vocab,vocab_to_id


def load_pretrain(glove_file,word_emb_dim):
    embedding_matrix,vocab = [], []
    with open(glove_file,encoding='utf-8',mode='r') as f:
        for i,line in enumerate(f):
            if i % 1e5 == 0:
                print('Current index is %d' %i)
            try:
                line_split = line.strip().split()
                if len(line_split) == word_emb_dim + 1:
                    # if line_split[0] in vocab_set:
                    vocab.append(line_split[0])
                    embedding_matrix.append([float(x) for x in line_split[1:]])
            except:
                pass
    vocab_to_id = {u:i for i,u in enumerate(vocab)}
    id_to_vocab = {v:u for u,v in vocab_to_id.items()}
    return embedding_matrix,vocab,vocab_to_id,id_to_vocab


def load_ED_data(filename,lower_case=False):
    """
    loading ner data, sentence and its corresponding word-level ner label
    """
    sents_all = []
    ners_all = []
    ner_1 = []
    ner_2 = []
    sent_tmp = []
    ner_tmp = []
    ner_1_tmp = []
    ner_2_tmp = []
    ner_vocab = set()
    doc_file_to_sents = {}
    with open(filename,encoding='utf-8',mode='r') as f:
        w_last = ''
        for line in f:
            line = line.strip()
            line_split = line.split(' ')
            if len(line_split) == 5:
                doc_file = line_split[1]
                if lower_case:
                    line_split[0] = str(line_split).lower()
                sent_tmp.append(line_split[0])

                ner_tmp.append(line_split[-1])
                ner_vocab.add(line_split[-1])
                ner_1_tmp_tmp = line_split[2]
                ner_1_tmp_tmp = ner_1_tmp_tmp
                ner_1_tmp.append(ner_1_tmp_tmp)
                ner_2_tmp_tmp = line_split[3]
                ner_2_tmp_tmp = ner_2_tmp_tmp
                ner_2_tmp.append(ner_2_tmp_tmp)
            else:
                if len(sent_tmp):
                    sents_all.append(sent_tmp)
                    ners_all.append(ner_tmp)
                    ner_1.append(ner_1_tmp)
                    ner_2.append(ner_2_tmp)
                sent_tmp = []
                ner_tmp = []
                ner_1_tmp = []
                ner_2_tmp = []
                if doc_file not in doc_file_to_sents:
                    doc_file_to_sents[doc_file] = [len(sents_all) - 1]
                else:
                    doc_file_to_sents[doc_file] += [len(sents_all) - 1]
            w_last = line_split[0]
        if len(sent_tmp) > 0:
            sents_all.append(sent_tmp)
            ners_all.append(ner_tmp)
            ner_1.append(ner_1_tmp)
            ner_2.append(ner_2_tmp)
            if doc_file not in doc_file_to_sents:
                doc_file_to_sents[doc_file] = [len(sents_all) - 1]
            else:
                doc_file_to_sents[doc_file] += [len(sents_all) - 1]
    return sents_all,ners_all,ner_vocab,ner_1,ner_2,doc_file_to_sents


def data_transformation_doc(sents_list,ner_1_list,ner_2_list,ner_list,vocab_2_id,ner_2_id,word_unk_id,ner_to_id_1,ner_to_id_2):
    """
    transform the raw data into numerics
    """
    encode_res = []
    for i,senti in enumerate(sents_list):
        neri = ner_list[i]
        ner_1_i = ner_1_list[i]
        ner_2_i = ner_2_list[i]
        ner_tmp = []
        sent_tmp = []
        ner_1_tmp = []
        ner_2_tmp = []
        for k, wordk in enumerate(senti):
            nerk = neri[k]
            try:
                sent_tmp.append(vocab_2_id[wordk])
            except:
                sent_tmp.append(word_unk_id)
            ner_tmp.append(ner_2_id[nerk])
            ner_1_tmp.append(ner_to_id_1[ner_1_i[k]])
            ner_2_tmp.append(ner_to_id_2[ner_2_i[k]])
        encode_res.append([sent_tmp,ner_1_tmp,ner_2_tmp,ner_tmp])
    return encode_res


def batch_generation_doc(doc_to_sents,enc_list,batch_size,max_doc_len,max_seq_len,vocab_2_id,ner_2_id, num_epoches=1):
    # padding and trimming
    ner_pad = ner_2_id['O']
    word_pad = vocab_2_id['<PAD>']
    valid_len_list = []
    for i,linei in enumerate(enc_list):
        senti = linei[0]
        ner_1_i = linei[1]
        ner_2_i = linei[2]
        neri = linei[3]
        valid_len_list.append(min(len(senti),max_seq_len))
        senti = senti[:max_seq_len]
        senti = senti + [word_pad] * max(0,max_seq_len-len(senti))
        neri = neri[:max_seq_len]
        neri = neri + [ner_pad] * max(0, max_seq_len - len(neri))
        ner_1_i = ner_1_i[:max_seq_len]
        ner_1_i = ner_1_i + [0] * max(0, max_seq_len - len(ner_1_i))
        ner_2_i = ner_2_i[:max_seq_len]
        ner_2_i = ner_2_i + [0] * max(0, max_seq_len - len(ner_2_i))
        enc_list[i] = [senti,ner_1_i,ner_2_i,neri]

    docs_all = []
    for kk,dockk in enumerate(list(doc_to_sents.keys())):
        sent_ids = doc_to_sents[dockk]
        if len(sent_ids) <= max_doc_len:
            sent_all = []
            ner_1_all = []
            ner_2_all = []
            ner_all = []
            valid_sents = len(sent_ids)
            valid_words = []
            for idi in sent_ids:
                sent_all.append(enc_list[idi][0])
                ner_1_all.append(enc_list[idi][1])
                ner_2_all.append(enc_list[idi][2])
                ner_all.append(enc_list[idi][3])
                valid_words.append(valid_len_list[idi])
            for kk in range(max_doc_len - valid_sents):
                sent_all.append(enc_list[idi][0])
                ner_1_all.append(enc_list[idi][1])
                ner_2_all.append(enc_list[idi][2])
                ner_all.append(enc_list[idi][3])
                valid_words.append(valid_len_list[idi])
            docs_all.append([sent_all,ner_1_all,ner_2_all,ner_all,valid_sents,valid_words])
        else:
            len_all = len(sent_ids)
            ndocs_mini = int(np.ceil(len_all / max_doc_len))
            for kk in range(ndocs_mini):
                init_step = kk * max_doc_len
                end_step = kk * max_doc_len + max_doc_len
                ids_tmp = sent_ids[init_step:end_step]
                sent_all = []
                ner_1_all = []
                ner_2_all = []
                ner_all = []
                valid_sents = len(ids_tmp)
                valid_words = []
                for idi in ids_tmp:
                    sent_all.append(enc_list[idi][0])
                    ner_1_all.append(enc_list[idi][1])
                    ner_2_all.append(enc_list[idi][2])
                    ner_all.append(enc_list[idi][3])
                    valid_words.append(valid_len_list[idi])
                for kk in range(max_doc_len - valid_sents):
                    sent_all.append(enc_list[idi][0])
                    ner_1_all.append(enc_list[idi][1])
                    ner_2_all.append(enc_list[idi][2])
                    ner_all.append(enc_list[idi][3])
                    valid_words.append(valid_len_list[idi])
                docs_all.append([sent_all, ner_1_all, ner_2_all, ner_all, valid_sents, valid_words])
    random.shuffle(docs_all)

    batches_all = []
    sent_alls = []
    ner_1_alls = []
    ner_2_alls = []
    ner_alls = []
    valid_sentss = []
    valid_wordss = []

    docs_all = docs_all * num_epoches

    for k,dock in enumerate(docs_all):
        if k % batch_size == 0 and k > 0:
            batches_all.append([sent_alls,ner_1_alls,ner_2_alls,ner_alls,batch_size,valid_sentss,valid_wordss])
            sent_alls = []
            ner_1_alls = []
            ner_2_alls = []
            ner_alls = []
            valid_sentss = []
            valid_wordss = []
            sent_alls.append(dock[0])
            ner_1_alls.append(dock[1])
            ner_2_alls.append(dock[2])
            ner_alls.append(dock[3])
            valid_sentss.append(dock[4])
            valid_wordss.append(dock[5])
        else:
            sent_alls.append(dock[0])
            ner_1_alls.append(dock[1])
            ner_2_alls.append(dock[2])
            ner_alls.append(dock[3])
            valid_sentss.append(dock[4])
            valid_wordss.append(dock[5])
    # paste the final
    len_valid = len(sent_alls)
    if len_valid == batch_size:
        batches_all.append([sent_alls, ner_1_alls, ner_2_alls, ner_alls, len_valid, valid_sentss, valid_wordss])
    else:
        sent_alls += [sent_alls[-1]] * (batch_size - len_valid)
        ner_1_alls += [ner_1_alls[-1]] * (batch_size - len_valid)
        ner_2_alls += [ner_2_alls[-1]] * (batch_size - len_valid)
        ner_alls += [ner_alls[-1]] * (batch_size - len_valid)
        valid_sentss += [valid_sentss[-1]] * (batch_size - len_valid)
        valid_wordss += [valid_wordss[-1]] * (batch_size - len_valid)
        batches_all.append([sent_alls, ner_1_alls, ner_2_alls, ner_alls, len_valid, valid_sentss, valid_wordss])
    return batches_all




if __name__ == "__main__":
    pass

