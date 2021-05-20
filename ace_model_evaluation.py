#!/usr/bin/env python
#-*- coding: utf-8 -*-


def ace_pred_result_stat(filename):
    wlast_true = ""
    wlast_pred = ""
    true_dict = set()
    pred_dict = set()
    id_true_init, id_true_end, id_pred_init, id_pred_end = 0, 0, 0, 0
    with open(filename,encoding='utf-8',mode='r') as f:
        for i,line in enumerate(f):
            line = line.strip()
            if len(line) > 0:
                line_split = line.split('\t')
                ## true label stats
                if line_split[1].startswith("B-"):
                    if wlast_true != '':  
                        true_dict.add('\t'.join([str(id_true_init), str(max(id_true_init, id_true_end)), wlast_true]))
                    id_true_init = i  # init id
                    id_true_end = i  # end id
                    wlast_true = line_split[1][2:]
                elif "I-" + wlast_true != line_split[1]:  # the last id is end of a trigger
                    if wlast_true != '':  # the last one is a trigger
                        true_dict.add('\t'.join([str(id_true_init), str(max(id_true_init, id_true_end)), wlast_true]))
                    wlast_true = ""
                elif "I-" + wlast_true == line_split[1]:  # the same with the last event type
                    id_true_end = i
                    wlast_true = line_split[1][2:]
                else:  # different from last label, and not start with B- 
                    if wlast_true != '':
                        true_dict.add('\t'.join([str(id_true_init), str(max(id_true_init, id_true_end)), wlast_true]))
                    wlast_true = ""

                ## pred label stats
                if line_split[2].startswith("B-"):
                    if wlast_pred != '':
                        pred_dict.add('\t'.join([str(id_pred_init), str(max(id_pred_init, id_pred_end)), wlast_pred]))
                    id_pred_init = i
                    id_pred_end = i
                    wlast_pred = line_split[2][2:]
                elif "I-" + wlast_pred != line_split[2]:  # begging of new trigger
                    if wlast_pred != '':
                        pred_dict.add('\t'.join([str(id_pred_init), str(max(id_pred_init, id_pred_end)), wlast_pred]))
                    wlast_pred = ""
                elif "I-" + wlast_pred == line_split[2]:
                    id_pred_end = i
                    wlast_pred = line_split[2][2:]
                else:
                    if wlast_pred != '':
                        pred_dict.add('\t'.join([str(id_pred_init), str(max(id_pred_init, id_pred_end)), wlast_pred]))
                    wlast_pred = ""
            else:
                if wlast_true != '':
                    true_dict.add('\t'.join([str(id_true_init), str(max(id_true_init, id_true_end)), wlast_true]))
                if wlast_pred != '':
                    pred_dict.add('\t'.join([str(id_pred_init), str(max(id_pred_init, id_pred_end)), wlast_pred]))
                wlast_true = ""
                wlast_pred = ""

    true_cnt = len(true_dict)
    pred_cnt = len(pred_dict)
    acc_cnt = len(pred_dict & true_dict)
    prec_tmp = acc_cnt / (pred_cnt + 1e-8)
    recall_tmp = acc_cnt / (true_cnt + 1e-8)
    f1_tmp = 2 * prec_tmp * recall_tmp / (prec_tmp + recall_tmp + 1e-8)
    return prec_tmp,recall_tmp,f1_tmp


def write_2_file(filename, ED_2_id, label_true_list,valid_len_list,words_sents, label_pred_list, id_2_vocab):
    id_to_ner_final = {v: u for u, v in ED_2_id.items()}
    with open(filename, encoding='utf-8', mode='w') as f:
        init_step = 0
        k = 0
        len_all = len(label_true_list)
        while init_step < len_all:
            end_step = init_step + valid_len_list[k]
            words_tmp = words_sents[init_step:end_step]
            pred_label_tmp_tmp = label_pred_list[init_step:end_step]
            true_label_tmp_tmp = label_true_list[init_step:end_step]
            for i in range(len(words_tmp)):
                f.write('\t'.join([id_2_vocab[words_tmp[i]],
                                   id_to_ner_final[true_label_tmp_tmp[i]],
                                   id_to_ner_final[pred_label_tmp_tmp[i]]]) + '\n')
            f.write('\n')
            init_step = end_step
            k += 1


if __name__ == "__main__":
    pass






