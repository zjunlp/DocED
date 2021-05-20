#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import time
import json
import random
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_integer('encode_h', 100, 'dim of encoding layer')
tf.flags.DEFINE_integer('decode_h', 200, 'dim of decoding layer')
tf.flags.DEFINE_integer('tag_dim', 100, 'dimension of tags')
tf.flags.DEFINE_integer('event_info_h', 100, 'hidden size of sentence level information aggregation layer')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('max_doc_len', 8, 'max number of sentences in a document')
tf.flags.DEFINE_integer('max_seq_len', 50, 'maximum length of sequence')
tf.flags.DEFINE_integer('num_tag_layers', 2, 'number of tagging layers')
tf.flags.DEFINE_integer('reverse_seq', 1, 'decoder mechanism')
tf.flags.DEFINE_string('tagging_mechanism', "backward_decoder", 'decoder mechanism')
tf.flags.DEFINE_integer('ner_dim_1', 20, 'embedding size of level-1 NER')
tf.flags.DEFINE_integer('ner_dim_2', 20, 'embedding size of level-2 NER')
tf.flags.DEFINE_integer('self_att_not', 1, 'self attention or not')
tf.flags.DEFINE_integer('context_info', 1,
                        '0: single sentence information, 1: information of two neighbor sentences')
tf.flags.DEFINE_float('penalty_coef', 2e-5, 'penalty coefficient')
tf.flags.DEFINE_float('event_vector_trans', 1, 'event_vector_trans')

tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epoches')
tf.flags.DEFINE_integer('eval_every_steps', 100, 'Number of epoches')
tf.flags.DEFINE_integer('num_epochs_warm', 0, 'Number of epoches of warm start')
tf.flags.DEFINE_integer('nconsect_epoch', 3, 'early stopping epoches')
tf.flags.DEFINE_float('weight_decay', 1, 'truncation of event attention weights')

tf.flags.DEFINE_float('warm_learning_rate', 1e-5, 'warm-up learning rate')
tf.flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.99, 'decay rate')

tf.flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate')
tf.flags.DEFINE_float('grad_clip', 10, 'grad clip to prevent gradient exlode')
tf.flags.DEFINE_float('positive_weights', 1, 'weights for positive sample')

tf.flags.DEFINE_string('train_file', './data-ACE/example_new.train', 'train file')
tf.flags.DEFINE_string('dev_file', './data-ACE/example_new.dev', 'dev file')
tf.flags.DEFINE_string('test_file', './data-ACE/example_new.test', 'test file')
tf.flags.DEFINE_string('embedding_file','./embedding/embeddings.txt','pretrained embedding file')
tf.flags.DEFINE_integer('word_emb_dim', 100, 'word embedding size')

tf.flags.DEFINE_string('NER_dict_file', './dict/event_types.txt', 'ner dict file')
tf.flags.DEFINE_string('ner_1_dict_file', './dict/ner_1.txt', 'level-1 ner dict file')
tf.flags.DEFINE_string('ner_2_dict_file', './dict/ner_2.txt', 'level-2 ner dict file')

FLAGS = tf.flags.FLAGS

lower_case = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.6


def train(seed_id=1):
    # set seed
    tf.set_random_seed(seed_id)

    from MLBiNet import MLBiNet

    from utils_init import load_ED_data
    from utils_init import data_transformation_doc
    from utils_init import batch_generation_doc

    from utils_init import load_vocab
    from utils_init import load_pretrain

    from ace_model_evaluation import write_2_file, ace_pred_result_stat

    with tf.Graph().as_default() as g:
        # loading the embedding matrix
        embedding_matrix, vocab_words, vocab_2_id, id_2_vocab = load_pretrain(FLAGS.embedding_file,
                                                                                    FLAGS.word_emb_dim)
        print('shape of embedding_matrix is:', np.asmatrix(embedding_matrix).shape)

        # load train, dev, test data
        sents_train, ners_train, ner_vocab, ner_1_train, ner_2_train, doc_file_to_sents_train = \
            load_ED_data(FLAGS.train_file, lower_case=lower_case)

        # load the vocab of event type
        _, ED_2_id = load_vocab(FLAGS.NER_dict_file)
        print("ner_2_id is:\t", ED_2_id)

        sents_dev, ners_dev, _, ner_1_dev, ner_2_dev, doc_file_to_sents_dev = \
            load_ED_data(FLAGS.dev_file, lower_case=lower_case)
        sents_test, ners_test, _, ner_1_test, ner_2_test, doc_file_to_sents_test = \
            load_ED_data(FLAGS.test_file, lower_case=lower_case)
        print("load_ner_data finished!")
        print("doc_file_to_sents_test:\t", doc_file_to_sents_test)

        # load NER label
        ner_vocab_1, ner_to_id_1 = load_vocab(FLAGS.ner_1_dict_file)
        ner_vocab_2, ner_to_id_2 = load_vocab(FLAGS.ner_2_dict_file)
        print("NER vocab loaded!")

        # encoding the train, dev, test data
        encode_train = data_transformation_doc(sents_train, ner_1_train, ner_2_train, ners_train,
                                               vocab_2_id, ED_2_id, vocab_2_id['<UNK>'], ner_to_id_1, ner_to_id_2)
        encode_dev = data_transformation_doc(sents_dev, ner_1_dev, ner_2_dev, ners_dev,
                                             vocab_2_id, ED_2_id, vocab_2_id['<UNK>'], ner_to_id_1, ner_to_id_2)
        encode_test = data_transformation_doc(sents_test, ner_1_test, ner_2_test, ners_test,
                                              vocab_2_id, ED_2_id, vocab_2_id['<UNK>'], ner_to_id_1, ner_to_id_2)
        print("Document data transformation finished!")

        # batch generating
        train_batches = batch_generation_doc(doc_file_to_sents_train, encode_train, FLAGS.batch_size, FLAGS.max_doc_len,
                                             FLAGS.max_seq_len, vocab_2_id, ED_2_id, num_epoches=FLAGS.num_epochs)
        dev_batches = batch_generation_doc(doc_file_to_sents_dev, encode_dev, FLAGS.batch_size, FLAGS.max_doc_len,
                                           FLAGS.max_seq_len, vocab_2_id, ED_2_id, num_epoches=1)
        test_batches = batch_generation_doc(doc_file_to_sents_test, encode_test, FLAGS.batch_size, FLAGS.max_doc_len,
                                            FLAGS.max_seq_len, vocab_2_id, ED_2_id, num_epoches=1)
        print("batch_generation_doc finished!")

        print('Begin model initialization!')
        with tf.Session(config=config_gpu) as sess:
            model = MLBiNet(
                encode_h = FLAGS.encode_h,
                decode_h = FLAGS.decode_h,
                tag_dim = FLAGS.tag_dim,
                event_info_h = FLAGS.event_info_h,
                word_emb_mat = np.array(embedding_matrix),
                batch_size = FLAGS.batch_size,
                max_doc_len = FLAGS.max_doc_len,
                max_seq_len = FLAGS.max_seq_len,
                id_O = ED_2_id['O'],
                num_tag_layers = FLAGS.num_tag_layers,
                weight_decay = FLAGS.weight_decay,
                reverse_seq = FLAGS.reverse_seq,
                class_size = len(ED_2_id),
                tagging_mechanism = FLAGS.tagging_mechanism,
                ner_size_1 = len(ner_to_id_1),
                ner_dim_1 = FLAGS.ner_dim_1,
                ner_size_2 = len(ner_to_id_2),
                ner_dim_2 = FLAGS.ner_dim_2,
                self_att_not = FLAGS.self_att_not,
                context_info = FLAGS.context_info,
                event_vector_trans = FLAGS.event_vector_trans
            )
            print('encoder-decoder model initialized!')

            loss_ed = model.loss
            for tvarsi in tf.trainable_variables():
                if tvarsi.name != 'word_emb_mat:0':
                    loss_ed += FLAGS.penalty_coef * tf.reduce_sum(tvarsi ** 2)
                else:
                    print("\n\n{} is not penalied!\n\n".format(tvarsi))

            with tf.name_scope('accuracy'):
                label_pred_naive = model.label_pred
                label_pred = model.label_pred
                label_true = model.label_true
                acc_cnt_naive = tf.reduce_sum(tf.cast(tf.equal(label_pred_naive,label_true),dtype=tf.float32))
                acc_cnt = tf.reduce_sum(tf.cast(tf.equal(label_pred,label_true),dtype=tf.float32))
                cnt_all = tf.reduce_sum(tf.cast(tf.greater(label_true,-1),dtype=tf.float32))
                acc_rate = acc_cnt / cnt_all

                valid_len_final = model.valid_len_list

            timestamp = str(int(time.time()))
            out_dir = os.path.join('./runs', timestamp)
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            param_dict = FLAGS.flag_values_dict()
            param_dict['lower_case'] = lower_case

            with open(os.path.join(checkpoint_dir,'config.json'), "w") as f:
                f.write(json.dumps(param_dict, indent=2, ensure_ascii=False))

            tvars = tf.trainable_variables()
            for kk, tvarsi in enumerate(tvars):
                print('The %d-th tvars is %s' % (kk, tvarsi))

            global_step = tf.Variable(0, trainable=False)

            learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=len(train_batches) // int(FLAGS.num_epochs),
                                                       decay_rate=FLAGS.decay_rate)
            tvars_no_emb = [x for x in tvars if 'word_emb_mat' not in x.name]
            opt_ed_NO_emb_sent = tf.train.AdamOptimizer(learning_rate)
            #
            grads_trig_sent_NO_EMB, _ = tf.clip_by_global_norm(tf.gradients(loss_ed, tvars_no_emb), FLAGS.grad_clip)
            grads_and_vars_trig_sent_NO_EMB = tuple(zip(grads_trig_sent_NO_EMB, tvars_no_emb))
            train_ed_NO_emb = opt_ed_NO_emb_sent.apply_gradients(grads_and_vars_trig_sent_NO_EMB, global_step=global_step)
            sess.run(tf.global_variables_initializer())

            def train_step(train_batch,epoch):
                positive_weights = FLAGS.positive_weights

                feed_dict = {
                    model.dropout_rate: FLAGS.dropout_rate,
                    model.input_docs: np.array(train_batch[0]),
                    model.ner_docs_1: np.array(train_batch[1]),
                    model.ner_docs_2: np.array(train_batch[2]),
                    model.input_label_docs: np.array(train_batch[3]),
                    model.valid_batch: train_batch[4],
                    model.valid_sent_len: np.array(train_batch[5]),
                    model.valid_words_len: np.array(train_batch[6]),
                    model.positive_weights: positive_weights
                }

                _, loss_trigger_tmp, acc_rate_tmp, step_curr = sess.run([train_ed_NO_emb, loss_ed, acc_rate, global_step],
                                                                        feed_dict)
                return loss_trigger_tmp,step_curr,acc_rate_tmp


            def dev_test_step(dev_batches):
                def dev_ont_batch(dev_batch):
                    feed_dict = {
                        model.dropout_rate: 0,
                        model.input_docs: np.array(dev_batch[0]),
                        model.ner_docs_1: np.array(dev_batch[1]),
                        model.ner_docs_2: np.array(dev_batch[2]),
                        model.input_label_docs: np.array(dev_batch[3]),
                        model.valid_batch: dev_batch[4],
                        model.valid_sent_len: np.array(dev_batch[5]),
                        model.valid_words_len: np.array(dev_batch[6]),
                        model.positive_weights: 1.0
                    }
                    acc_cnt_tmp, cnt_all_tmp,acc_cnt_naive_tmp,valid_len_tmp,\
                    label_pred_tmp, label_pred_naive_tmp, label_true_tmp, final_words_id_tmp, loss_tmp \
                        = sess.run([acc_cnt,cnt_all,acc_cnt_naive,valid_len_final,label_pred, label_pred_naive,
                                    label_true,model.final_words_id,loss_ed], feed_dict)
                    return acc_cnt_tmp, cnt_all_tmp,acc_cnt_naive_tmp,valid_len_tmp,label_pred_tmp, \
                           label_pred_naive_tmp, label_true_tmp,final_words_id_tmp, loss_tmp

                acc_cnt_list, cnt_all_list = [], []
                acc_cnt_naive_list, cnt_all_naive_list = [], []
                label_pred_list, label_pred_naive_list, label_true_list = [],[],[]
                valid_len_list = []
                words_sents = []
                loss_dev_test = 0
                len_seq_all = 0
                for dev_batchi in dev_batches:
                    acc_cnt_tmp, cnt_all_tmp,acc_cnt_naive_tmp,valid_len_tmp,\
                    label_pred_tmp, label_pred_naive_tmp, label_true_tmp,final_words_id_tmp, loss_tmp_i\
                        = dev_ont_batch(dev_batchi)
                    acc_cnt_list.append(acc_cnt_tmp)
                    cnt_all_list.append(cnt_all_tmp)
                    acc_cnt_naive_list.append(acc_cnt_naive_tmp)
                    label_pred_list.extend(label_pred_tmp)
                    label_pred_naive_list.extend(label_pred_naive_tmp)
                    label_true_list.extend(label_true_tmp)
                    valid_len_list.extend(valid_len_tmp)
                    words_sents.extend(final_words_id_tmp)
                    loss_dev_test += loss_tmp_i * len(label_pred_naive_tmp)
                    len_seq_all += len(label_pred_naive_tmp)
                loss_dev_test = loss_dev_test / (len_seq_all + 1e-8)

                prec_dev = sum(acc_cnt_list) / sum(cnt_all_list)
                prec_dev_naive = sum(acc_cnt_naive_list) / sum(cnt_all_list)
                return prec_dev,prec_dev_naive,words_sents,label_pred_list,\
                       label_true_list,valid_len_list,loss_dev_test

            print('Total train batch is:\t',len(train_batches),flush=True)

            prec_test_best = 0
            loss_dev_best = 10000
            loss_dev_second = 10000
            loss_dev_list = []
            nconsect = 0
            print("total train steps:\t",  len(train_batches))
            for i, train_batchi in enumerate(train_batches):
                epoch = i // FLAGS.eval_every_steps
                loss_trigger_tmp, step_curr, acc_rate_tmp = train_step(train_batchi,0)
                if i % 1e1 == 0:
                    print('epoch {}, step: {},loss: {},acc_rate: {}'.format(
                        epoch,step_curr,loss_trigger_tmp,acc_rate_tmp), flush=True)

                if i % FLAGS.eval_every_steps == 0 or i == len(train_batches) - 1:
                    prec_dev,prec_dev_naive,words_sents,label_pred_list,\
                        label_true_list,valid_len_list,loss_dev_ = dev_test_step(dev_batches)
                    print('epoch {} prec_dev is: \n'.format(epoch), prec_dev, flush=True)
                    if epoch == 0:
                        os.makedirs(os.path.join(checkpoint_dir, 'dev'))
                    filename_dev = os.path.join(checkpoint_dir, 'dev/test_result_{}.txt').format(step_curr)
                    write_2_file(filename_dev, ED_2_id, label_true_list,valid_len_list,
                                 words_sents, label_pred_list, id_2_vocab)
                    prec_event_dev, recall_event_dev, f1_event_dev = ace_pred_result_stat(filename_dev)
                    print('epoch: {}, loss_dev_: {}'.format(epoch, loss_dev_), flush=True)
                    print('epoch: {}, prec_event_dev: {}, recall_event_dev: {}, f1_event_dev: {}'.format(
                        epoch, prec_event_dev, recall_event_dev, f1_event_dev), flush=True)

                    loss_dev_list.append(loss_dev_)
                    loss_dev_list = sorted(loss_dev_list,key=lambda x: x, reverse=False)
                    if len(loss_dev_list) > 2:
                        loss_dev_second = loss_dev_list[2]
                    if loss_dev_ > loss_dev_best:
                        if loss_dev_ > loss_dev_second:
                            nconsect += 1
                        else:
                            nconsect = 0
                    else:
                        nconsect = 0
                        loss_dev_best = loss_dev_

                    print('\n')
                    prec_test,prec_test_naive,words_sents,label_pred_list,\
                    label_true_list,valid_len_list, loss_test_ = dev_test_step(test_batches)
                    print('epoch {} prec_test is: \n'.format(epoch), prec_test, flush=True)
                    print('\n')
                    # write to file
                    if epoch == 0:
                        os.makedirs(os.path.join(checkpoint_dir, 'test'))
                    filename_test = os.path.join(checkpoint_dir, 'test/test_result_{}.txt').format(step_curr)
                    write_2_file(filename_test, ED_2_id, label_true_list,valid_len_list,
                                 words_sents, label_pred_list, id_2_vocab)
                    prec_event_test, recall_event_test, f1_event_test = ace_pred_result_stat(filename_test)
                    print('epoch: {}, prec_event_test: {}, recall_event_test: {}, f1_event_test:{}'.format(
                        epoch, prec_event_test, recall_event_test, f1_event_test), flush=True)

                    if prec_test_best < f1_event_test:
                        prec_test_best = f1_event_test

                    print('The best dev loss value is:\t', [loss_dev_best,nconsect])
                    # print('The best dev f1 value is:\t', [prec_dev_best,nconsect])
                    print('The best test f1 value is:\t', prec_test_best)
                    with open(os.path.join(checkpoint_dir, 'test_result.txt'), encoding='utf-8', mode='a') as f:
                        f.write('\t'.join([str(epoch), str(prec_event_test),str(recall_event_test),
                                           str(f1_event_test), str(loss_dev_best), str(loss_dev_second), str(nconsect)]) + '\n')

                    if nconsect >= FLAGS.nconsect_epoch:
                        break
    tf.reset_default_graph()

if __name__ == "__main__":
    # train()
    pass