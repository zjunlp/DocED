#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn


class MLBiNet:
    def __init__(self,
                 encode_h,  # hidden size of sentence encoding
                 decode_h,  # hidden size of sentence decoding
                 tag_dim,  # hidden size of event tag
                 event_info_h,  # hidden size of event info integration model
                 word_emb_mat,  # word embedding matrix
                 batch_size,  # batch size
                 max_doc_len,  # max length of doc
                 max_seq_len,  # max length of sequence
                 id_O,  # location of other event / negative event
                 num_tag_layers,  # number of tagging layers
                 weight_decay,  # weight decay of each tagging layer
                 reverse_seq,  # reverse the sequence or not when aggregating information of next sentence
                 class_size,  # class size
                 tagging_mechanism="bidirectional_decoder",  # forward_decoder, backward_decoder, bidirectional_decoder
                 ner_size_1=None,  # size of level-1 ner vocab
                 ner_dim_1=None,  # dimension of level-1 ner embedding
                 ner_size_2=None,  # size of level-2 ner vocab
                 ner_dim_2=None,  # dimension of level-2 ner embedding
                 self_att_not=1,  # concat word embedding or not
                 context_info=1,  # 0: single sentence information, 1: information of two neighbor sentences
                 event_vector_trans=1 # nonlinear transformation for the event vector
                 ):
        self.encode_h = encode_h
        self.decode_h = decode_h
        self.tag_dim = tag_dim
        self.event_info_h = event_info_h
        self.word_emb_mat = word_emb_mat
        self.batch_size = batch_size
        self.max_doc_len = max_doc_len
        self.max_seq_len = max_seq_len
        self.id_O = id_O
        self.num_tag_layers = num_tag_layers
        self.weight_decay = weight_decay
        self.reverse_seq = reverse_seq
        self.class_size = class_size
        self.tagging_mechanism = tagging_mechanism

        self.ner_size_1 = ner_size_1
        self.ner_dim_1 = ner_dim_1
        self.ner_size_2 = ner_size_2
        self.ner_dim_2 = ner_dim_2
        self.self_att_not = self_att_not

        self.context_info = context_info
        self.event_vector_trans = event_vector_trans

        # global initializer
        self.initializer = initializers.xavier_initializer()

        # initialize the word embedding matrix
        self.word_emb_mat = tf.cast(self.word_emb_mat, dtype=tf.float32)
        self.word_embedding_init()

        # placeholders
        self.input_docs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,
                                                                self.max_doc_len, self.max_seq_len], name='input_docs')
        self.ner_docs_1 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,
                                                                self.max_doc_len, self.max_seq_len], name='ner_docs_1')
        self.ner_docs_2 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,
                                                                self.max_doc_len, self.max_seq_len], name='ner_docs_2')
        self.input_label_docs = tf.placeholder(dtype=tf.int32,
                                               shape=[self.batch_size, self.max_doc_len, self.max_seq_len],
                                               name='input_label_docs')
        self.valid_batch = tf.placeholder(dtype=tf.int32, shape=(), name='valid_batch')
        self.valid_sent_len = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='valid_sent_len')
        self.valid_words_len = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_doc_len],
                                              name='valid_words_len')
        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_rate')
        self.positive_weights = tf.placeholder(dtype=tf.float32, shape=(), name='positive_weights')

        # embedding layer
        self.word_embedding_lookup = self.embedding_layer()

        # [unk] event and semantic information aggregation embedding
        self.unk_event_semantic = tf.Variable(tf.truncated_normal(shape=[1, self.event_info_h], stddev=0.1),
                                              trainable=True, name="unk_event_semantic")
        # self.unk_event_semantic = tf.zeros(shape=[1,self.event_info_h])

        # sentence encoding layer
        emb_size_curr = self.word_embedding_lookup.get_shape().as_list()[-1]
        self.lstm_inputs = tf.nn.dropout(self.word_embedding_lookup, keep_prob=1 - self.dropout_rate)

        print("embedding dimension before encoding layer:\t", emb_size_curr)

        words_enc, _, _ = self.sent_encode_layer(
            tf.reshape(self.lstm_inputs, [self.batch_size * self.max_doc_len,
                                          self.max_seq_len, emb_size_curr]),
            tf.reshape(self.valid_words_len, shape=[-1]), name='sent_enc_model')

        print("embedding dimension after encoding layer:\t", words_enc.get_shape().as_list()[-1])

        # self-attention
        words_enc = tf.reshape(words_enc, [self.batch_size, self.max_doc_len, self.max_seq_len, -1])
        if self.self_att_not:
            words_enc = self.sent_self_att(words_enc, self.valid_words_len)

        print("embedding dimension after self-attention:\t", words_enc.get_shape().as_list()[-1])

        # concat with looking up embedding
        words_enc = tf.concat([words_enc, self.word_embedding_lookup], axis=-1)
        words_enc = tf.nn.dropout(words_enc, keep_prob=1 - self.dropout_rate)

        print("embedding dimension before decoding:\t", words_enc.get_shape().as_list()[-1])

        # mask all padding vectors
        dim_curr = words_enc.get_shape().as_list()[-1]
        mask_padding_ind = tf.sequence_mask(self.valid_words_len, maxlen=self.max_seq_len, dtype=tf.float32)
        self.mask_padding_ind = tf.tile(tf.expand_dims(mask_padding_ind, axis=3), multiples=[1, 1, 1, dim_curr])

        self.words_enc = words_enc * self.mask_padding_ind

        # tagging via multi-tagging network
        if self.tagging_mechanism == "forward_decoder":
            tag_vect, tag_vect_layerwise = self.forward_cross_sent_ED(words_enc=self.words_enc, tag_dim=self.tag_dim,
                                                                      num_tag_layers=self.num_tag_layers,
                                                                      weight_decay=self.weight_decay)
        elif self.tagging_mechanism == "backward_decoder":
            tag_vect, tag_vect_layerwise = self.backward_cross_sent_ED(words_enc=self.words_enc, tag_dim=self.tag_dim,
                                                                       num_tag_layers=self.num_tag_layers,
                                                                       weight_decay=self.weight_decay)
        elif self.tagging_mechanism == "bidirectional_decoder":
            tag_vect_fw, tag_vect_bw, tag_vect_lw_fw, tag_vect_lw_bw = self.biderectional_cross_sent_ED(
                words_enc=self.words_enc, tag_dim=self.tag_dim, num_tag_layers=self.num_tag_layers,
                weight_decay=self.weight_decay)
            tag_vect = tf.concat([tag_vect_fw, tag_vect_bw], axis=-1)
            tag_vect_layerwise = tf.concat([tag_vect_lw_fw, tag_vect_lw_bw], axis=-1)
        elif self.tagging_mechanism == "agg_average":
            tag_vect_fw, tag_vect_bw, tag_vect_lw_fw, tag_vect_lw_bw = self.agg_choice_cross_sent_ED(
                                            words_enc=self.words_enc,
                                            tag_dim=self.tag_dim,
                                            num_tag_layers=self.num_tag_layers,
                                            weight_decay=self.weight_decay,
                                            agg_choice="average")
            tag_vect = tf.concat([tag_vect_fw, tag_vect_bw], axis=-1)
            tag_vect_layerwise = tf.concat([tag_vect_lw_fw, tag_vect_lw_bw], axis=-1)
        elif self.tagging_mechanism == "agg_concat":
            tag_vect_fw, tag_vect_bw, tag_vect_lw_fw, tag_vect_lw_bw = self.agg_choice_cross_sent_ED(
                                            words_enc=self.words_enc,
                                            tag_dim=self.tag_dim,
                                            num_tag_layers=self.num_tag_layers,
                                            weight_decay=self.weight_decay,
                                            agg_choice="concat")
            tag_vect = tf.concat([tag_vect_fw, tag_vect_bw], axis=-1)
            tag_vect_layerwise = tf.concat([tag_vect_lw_fw, tag_vect_lw_bw], axis=-1)
        else:
            print("tagging_mechanism assigned is not supported!")

        # self loss function
        self.loss, self.label_true, self.label_pred, self.valid_len_list = self.loss_layer(tag_vect)

    def word_embedding_init(self):
        """
        initialize the word embedding matrix
        """
        if self.word_emb_mat is None:
            print("The embedding matrix must be initialized!")
        else:
            self.word_emb_mat = tf.Variable(self.word_emb_mat, trainable=True, name='word_emb_mat')

    def embedding_layer(self):
        """
        embedding layer with respect to the word embedding matrix
        """
        embedding_tmp = tf.nn.embedding_lookup(self.word_emb_mat, self.input_docs)
        # looking up the level-1 ner embedding
        if self.ner_size_1 is not None:
            ner_mat_1 = tf.get_variable(name="ner_mat_1", shape=[self.ner_size_1, self.ner_dim_1],
                                        dtype=tf.float32, initializer=self.initializer)
            emb_ner1_tmp = tf.nn.embedding_lookup(ner_mat_1, self.ner_docs_1)
            embedding_tmp = tf.concat([embedding_tmp, emb_ner1_tmp], axis=-1)
        # looking up the level-2 ner embedding
        if self.ner_size_2 is not None:
            ner_mat_2 = tf.get_variable(name="ner_mat_2", shape=[self.ner_size_2, self.ner_dim_2],
                                        dtype=tf.float32, initializer=self.initializer)
            emb_ner2_tmp = tf.nn.embedding_lookup(ner_mat_2, self.ner_docs_1)
            embedding_tmp = tf.concat([embedding_tmp, emb_ner2_tmp], axis=-1)
        return embedding_tmp

    def sent_encode_layer(self, embedding_input, valid_len, name):
        """
        sentence encoding layer to get representation of each words
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        self.encode_h,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True
                    )
            (outputs,
             (encoder_fw_final_state,
              encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                inputs=embedding_input,
                dtype=tf.float32,
                sequence_length=valid_len
            )
        words_out = tf.concat(outputs, axis=-1)
        final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), -1)
        final_state_add = (encoder_fw_final_state.h + encoder_bw_final_state.h) / 2
        return words_out, final_state, final_state_add

    def sent_self_att(self, words_enc, valid_words_len):
        """
        sentence-level self-attention
        :param words_enc: batch_size * max_doc_size * max_seq_len * dim
        :param valid_words_len: batch_size * max_doc_size
        """
        enc_dim_tmp = words_enc.get_shape().as_list()[-1]
        words_enc_new0 = tf.reshape(words_enc, [self.batch_size * self.max_doc_len, self.max_seq_len, enc_dim_tmp])
        valid_words_len_new = tf.reshape(valid_words_len, shape=[-1])

        def self_att(variable_scope="attention", weight_name="att_W"):
            """
            sentence level self attention with different window size
            """
            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                W = tf.get_variable(weight_name,
                                    shape=[enc_dim_tmp, enc_dim_tmp],
                                    dtype=tf.float32,
                                    initializer=self.initializer,
                                    )
                # x'Wx
                words_enc_new = tf.reshape(words_enc,
                                           [self.batch_size * self.max_doc_len * self.max_seq_len, enc_dim_tmp])
                words_enc_new = tf.matmul(words_enc_new, W)
                words_enc_new = tf.reshape(words_enc_new,
                                           [self.batch_size * self.max_doc_len, self.max_seq_len, enc_dim_tmp])
                # tanh(x'Wx)
                logit_self_att = tf.matmul(words_enc_new, tf.transpose(words_enc_new0, perm=[0, 2, 1]))
                logit_self_att = tf.tanh(logit_self_att)
                probs = tf.nn.softmax(logit_self_att)

                # mask invalid words
                mask_words = tf.sequence_mask(valid_words_len_new, maxlen=self.max_seq_len,
                                              dtype=tf.float32)  # 160 * 100
                mask_words = tf.tile(tf.expand_dims(mask_words, axis=1),
                                     multiples=[1, self.max_seq_len, 1])  # 160 * 100 * 100
                probs = probs * mask_words
                probs = tf.matmul(tf.matrix_diag(1 / (tf.reduce_sum(probs, axis=-1) + 1e-8)),
                                  probs)  # re-standardize the probability
                # attention output
                att_output = tf.matmul(probs, words_enc_new0)
                att_output = tf.reshape(att_output,
                                        shape=[self.batch_size, self.max_doc_len, self.max_seq_len, enc_dim_tmp])
            return att_output

        att_output = self_att(variable_scope="attention", weight_name="att_W")
        return att_output

    def info_agg_layer(self, pred_tag_vect, reverse_seq=False):
        """
        sentence-level event and semantic information aggregation layer
        """
        dim_curr = pred_tag_vect.get_shape().as_list()[-1]

        # mask invalid words
        mask_padding_ind = tf.sequence_mask(self.valid_words_len, maxlen=self.max_seq_len, dtype=tf.float32)
        mask_padding_ind = tf.tile(tf.expand_dims(mask_padding_ind, axis=3), multiples=[1, 1, 1, dim_curr])
        pred_tag_vect = pred_tag_vect * mask_padding_ind

        # reverse the sequence
        if reverse_seq:
            pred_tag_vect = pred_tag_vect[:, :, ::-1, :]
            var_name = "reversed_sent_info_agg_layer"
        else:
            var_name = "sent_info_agg_layer"

        info_agg_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.event_info_h, forget_bias=0.0, state_is_tuple=True,
                                                          name=var_name, reuse=tf.AUTO_REUSE)
        info_agg_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(info_agg_lstm_cell, output_keep_prob=1 - self.dropout_rate)
        # todo, change to bidirectional_dynamic_rnn
        # _, _, sent_event_sematic_info = self.sent_encode_layer(
        #     embedding_input=tf.reshape(pred_tag_vect, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1]),
        #     valid_len=tf.reshape(self.valid_words_len, [-1]),
        #     name=var_name
        # )
        _, (_, sent_event_sematic_info) = tf.nn.dynamic_rnn(cell=info_agg_lstm_cell,
                                                            inputs=tf.reshape(pred_tag_vect,
                                                                              shape=[self.batch_size * self.max_doc_len,
                                                                                     self.max_seq_len, -1]),
                                                            sequence_length=tf.reshape(self.valid_words_len, [-1]),
                                                            dtype=tf.float32
                                                            )
        sent_event_sematic_info = tf.reshape(sent_event_sematic_info,
                                             shape=[self.batch_size, self.max_doc_len, -1])
        return sent_event_sematic_info

    def info_agg_layer_bi(self, pred_tag_vect, reverse_seq=False):
        """
        sentence-level event and semantic information aggregation layer
        """
        dim_curr = pred_tag_vect.get_shape().as_list()[-1]

        # mask invalid words
        mask_padding_ind = tf.sequence_mask(self.valid_words_len, maxlen=self.max_seq_len, dtype=tf.float32)
        mask_padding_ind = tf.tile(tf.expand_dims(mask_padding_ind, axis=3), multiples=[1, 1, 1, dim_curr])
        pred_tag_vect = pred_tag_vect * mask_padding_ind

        # reverse the sequence
        if reverse_seq:
            pred_tag_vect = pred_tag_vect[:, :, ::-1, :]
            var_name = "reversed_sent_info_agg_layer"
        else:
            var_name = "sent_info_agg_layer"

        # info_agg_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.event_info_h, forget_bias=0.0, state_is_tuple=True,
        #                                                   name=var_name, reuse=tf.AUTO_REUSE)
        # info_agg_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(info_agg_lstm_cell, output_keep_prob=1 - self.dropout_rate)
        # todo, change to bidirectional_dynamic_rnn
        _, _, sent_event_sematic_info = self.sent_encode_layer(
            embedding_input=tf.reshape(pred_tag_vect, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1]),
            valid_len=tf.reshape(self.valid_words_len, [-1]),
            name=var_name
        )
        # _, (_, sent_event_sematic_info) = tf.nn.dynamic_rnn(cell=info_agg_lstm_cell,
        #                                                     inputs=tf.reshape(pred_tag_vect,
        #                                                                       shape=[self.batch_size * self.max_doc_len,
        #                                                                              self.max_seq_len, -1]),
        #                                                     sequence_length=tf.reshape(self.valid_words_len, [-1]),
        #                                                     dtype=tf.float32
        #                                                     )
        sent_event_sematic_info = tf.reshape(sent_event_sematic_info,
                                             shape=[self.batch_size, self.max_doc_len, -1])
        return sent_event_sematic_info

    def project(self, h_state, lstm_dim):
        """
        project the output of decoder model to a tag vector
        """
        enc_dim = h_state.get_shape().as_list()[-1]
        with tf.variable_scope("tag_project_layer", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W",
                                shape=[enc_dim, lstm_dim],
                                dtype=tf.float32,
                                initializer=self.initializer,
                                )
            b = tf.get_variable("b",
                                shape=[lstm_dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer()
                                )
            y_pre = tf.add(tf.matmul(h_state, W), b)
            tag_pre = tf.cast(tf.argmax(tf.nn.softmax(y_pre), axis=-1), tf.float32)
            return y_pre, tag_pre

    def forward_cross_sent_ED(self, words_enc, tag_dim, num_tag_layers, weight_decay):
        """
        forward-wise cross-sentence event tag event detection, modeling the forward-wise event correlation
        :param words_enc: words encoding
        :param num_tag_layers: number of tagging layers
        :param weight_decay: weight decay of tagging vectors of different layers
        """
        # decoding layer
        # all layers share the same decoder layer
        # for the first decoder layer, we set c_{i-1} and c_{i+1} with unk_event_semantic
        lstm_outputs = tf.reshape(words_enc, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                 name="forward_rnn_decoder", reuse=tf.AUTO_REUSE)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - self.dropout_rate)

        # mutli-tagging block
        tag_final = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_list = []

        init_state = lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        # event and semantic information of the previous sentence and next sentence sentence
        info_event_sem_pre_sent = tf.tile(self.unk_event_semantic,
                                          multiples=[self.batch_size * self.max_doc_len, 1])
        info_event_sem_next_sent = tf.tile(self.unk_event_semantic,
                                           multiples=[self.batch_size * self.max_doc_len, 1])

        # event and semantic information of the beginning sentence
        info_event_sem_init_sent = tf.tile(self.unk_event_semantic, multiples=[self.batch_size, 1])
        info_event_sem_init_sent = tf.expand_dims(info_event_sem_init_sent, axis=1)
        info_event_sem_mat0 = tf.tile(tf.expand_dims(self.unk_event_semantic, axis=0),
                                      multiples=[self.batch_size, self.max_doc_len, 1])
        with tf.variable_scope("forward_rnn_decoding_layer", reuse=tf.AUTO_REUSE):
            for layer_id in range(num_tag_layers):
                # initialize for each layer
                c_state, h_state = init_state
                tag_pre = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                tag_outputs = []
                for time_step in range(self.max_seq_len):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    if self.num_tag_layers > 1:
                        two_info = tf.concat([info_event_sem_pre_sent, info_event_sem_next_sent], axis=-1)
                        input_all = tf.concat([lstm_outputs[:, time_step, :], two_info, tag_pre], axis=-1)
                    else:
                        input_all = tf.concat([lstm_outputs[:, time_step, :], tag_pre], axis=-1)
                    (cell_output, (c_state, h_state)) = lstm_cell(input_all, (c_state, h_state))
                    tag_pre, tag_result = self.project(cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_pre = tf.tanh(tag_pre)
                    tag_outputs.append(tag_pre)
                tag_outputs = tf.reshape(tf.transpose(tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                self.max_seq_len, tag_dim])
                if self.num_tag_layers > 1:
                    # info aggregation of current sentence, [batch_size, max_doc_len,event_info_h]
                    info_event_sem_current_sent = self.info_agg_layer(tag_outputs, reverse_seq=False)

                    # corresponds to the information of previous sentence
                    info_event_sem_pre_sent = tf.concat([info_event_sem_init_sent,
                                                         info_event_sem_current_sent[:, :-1, :]], axis=1)
                    info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                         shape=[self.batch_size * self.max_doc_len, -1])

                    # find valid sentence firstly, and replace with emebedding of unk
                    info_event_sem_current_sent_bw = self.info_agg_layer(tag_outputs, reverse_seq=self.reverse_seq)

                    valid_sent_ind = tf.sequence_mask(self.valid_sent_len, maxlen=self.max_doc_len, dtype=tf.float32)
                    valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=2), multiples=[1, 1, self.event_info_h])
                    info_event_sem_current_sent_bw = info_event_sem_current_sent_bw * valid_sent_ind + \
                                                     info_event_sem_mat0 * (1 - valid_sent_ind)

                    # corresponds to the information of previous sentence
                    info_event_sem_next_sent = tf.concat([info_event_sem_current_sent_bw[:, 1:, :], info_event_sem_init_sent],
                                                         axis=1)
                    info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                          shape=[self.batch_size * self.max_doc_len, -1])

                tag_final += weight_decay ** layer_id * tag_outputs
                tag_final_list.append(tag_outputs)
        return tag_final, tag_final_list


    def backward_cross_sent_ED(self, words_enc, tag_dim, num_tag_layers, weight_decay):
        """
        backward-wise cross-sentence event tag event detection, modeling the backward-wise event correlation
        """
        # reshape the inputs and reverse it to cater to backward event extraction
        lstm_outputs = tf.reshape(words_enc, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1])
        lstm_outputs = lstm_outputs[:, ::-1, :]

        # decoding layer
        # all layers share the same decoder layer
        # for the first decoder layer, we set c_{i-1} and c_{i+1} with unk_event_semantic
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                 name="backward_rnn_decoder", reuse=tf.AUTO_REUSE)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - self.dropout_rate)

        # mutli-tagging block
        tag_final = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_list = []

        init_state = lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        # event and semantic information of the previous sentence and next sentence sentence
        info_event_sem_pre_sent = tf.tile(self.unk_event_semantic,
                                          multiples=[self.batch_size * self.max_doc_len, 1])
        info_event_sem_next_sent = tf.tile(self.unk_event_semantic,
                                           multiples=[self.batch_size * self.max_doc_len, 1])

        # event and semantic information of the final sentence
        info_event_sem_init_sent = tf.tile(self.unk_event_semantic, multiples=[self.batch_size, 1])
        info_event_sem_init_sent = tf.expand_dims(info_event_sem_init_sent, axis=1)
        info_event_sem_mat0 = tf.tile(tf.expand_dims(self.unk_event_semantic, axis=0),
                                      multiples=[self.batch_size, self.max_doc_len, 1])

        with tf.variable_scope("backward_rnn_decoding_layer", reuse=tf.AUTO_REUSE):
            for layer_id in range(num_tag_layers):
                # initialize for each layer
                c_state, h_state = init_state
                tag_next = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                tag_outputs = []
                for time_step in range(self.max_seq_len):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    if self.num_tag_layers > 1:
                        two_info = tf.concat([info_event_sem_pre_sent, info_event_sem_next_sent], axis=-1)
                        input_all = tf.concat([lstm_outputs[:, time_step, :], two_info, tag_next], axis=-1)
                    else:
                        input_all = tf.concat([lstm_outputs[:, time_step, :], tag_next], axis=-1)
                    (cell_output, (c_state, h_state)) = lstm_cell(input_all, (c_state, h_state))
                    tag_next, tag_result = self.project(cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_next = tf.tanh(tag_next)
                    tag_outputs.append(tag_next)
                tag_outputs = tf.reshape(tf.transpose(tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                self.max_seq_len, tag_dim])
                # recover the tag_outputs in order
                tag_outputs = tag_outputs[:, :, ::-1, :]

                if self.num_tag_layers > 1:
                    # info aggregation of current sentence, [batch_size, max_doc_len,event_info_h]
                    info_event_sem_current_sent = self.info_agg_layer(tag_outputs, reverse_seq=self.reverse_seq)

                    # find valid sentence firstly, and replace with emebedding of unk
                    valid_sent_ind = tf.sequence_mask(self.valid_sent_len, maxlen=self.max_doc_len, dtype=tf.float32)
                    valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=2), multiples=[1, 1, self.event_info_h])
                    info_event_sem_current_sent = info_event_sem_current_sent * valid_sent_ind + \
                                                  info_event_sem_mat0 * (1 - valid_sent_ind)

                    # corresponds to the information of previous sentence
                    info_event_sem_next_sent = tf.concat([info_event_sem_current_sent[:, 1:, :], info_event_sem_init_sent],
                                                         axis=1)
                    info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                          shape=[self.batch_size * self.max_doc_len, -1])

                    # information of previous sentence, [batch_size, max_doc_len,event_info_h]
                    info_event_sem_current_sent = self.info_agg_layer(tag_outputs, reverse_seq=False)
                    info_event_sem_pre_sent = tf.concat([info_event_sem_init_sent, info_event_sem_current_sent[:, :-1, :]],
                                                        axis=1)
                    info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                         shape=[self.batch_size * self.max_doc_len, -1])

                tag_final += weight_decay ** layer_id * tag_outputs
                tag_final_list.append(tag_outputs)
        return tag_final, tag_final_list


    def biderectional_cross_sent_ED(self, words_enc, tag_dim, num_tag_layers, weight_decay):
        """
        birectional cross-sentence event tag event detection, modeling birectional event correlation
        """
        # decoding layer
        # all layers share the same decoder layer
        # for the first decoder layer, we set c_{i-1} and c_{i+1} with unk_event_semantic
        lstm_outputs = tf.reshape(words_enc, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1])
        backward_lstm_outputs = lstm_outputs[:, ::-1, :]

        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                    name="forward_rnn_decoder", reuse=tf.AUTO_REUSE)
        fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=1 - self.dropout_rate)

        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                    name="backward_rnn_decoder", reuse=tf.AUTO_REUSE)
        bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=1 - self.dropout_rate)

        # mutli-tagging block
        tag_final_fw = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_bw = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_list_fw = []
        tag_final_list_bw = []

        fw_init_state = fw_lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        bw_init_state = bw_lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        # event and semantic information of the previous sentence and next sentence sentence
        info_event_sem_pre_sent = tf.tile(self.unk_event_semantic,
                                          multiples=[self.batch_size * self.max_doc_len, 1])
        info_event_sem_next_sent = tf.tile(self.unk_event_semantic,
                                           multiples=[self.batch_size * self.max_doc_len, 1])

        # event and semantic information of the beginning sentence
        info_event_sem_init_sent = tf.tile(self.unk_event_semantic, multiples=[self.batch_size, 1])
        info_event_sem_init_sent = tf.expand_dims(info_event_sem_init_sent, axis=1)
        info_event_sem_mat0 = tf.tile(tf.expand_dims(self.unk_event_semantic, axis=0),
                                      multiples=[self.batch_size, self.max_doc_len, 1])

        with tf.variable_scope("bidirectional_rnn_decoding_layer", reuse=tf.AUTO_REUSE):
            for layer_id in range(num_tag_layers):
                # initialize for each layer
                fw_c_state, fw_h_state = fw_init_state
                bw_c_state, bw_h_state = bw_init_state
                tag_fw = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                tag_bw = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                fw_tag_outputs = []
                bw_tag_outputs = []
                for time_step in range(self.max_seq_len):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    # concat two event information
                    if self.num_tag_layers > 1:
                        if not self.context_info:
                            fw_input_all = tf.concat([lstm_outputs[:, time_step, :], info_event_sem_pre_sent, tag_fw],
                                                     axis=-1)
                            bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :],
                                                      info_event_sem_next_sent, tag_bw], axis=-1)
                        else:
                            two_info = tf.concat([info_event_sem_pre_sent, info_event_sem_next_sent], axis=-1)
                            fw_input_all = tf.concat([lstm_outputs[:, time_step, :], two_info, tag_fw], axis=-1)
                            bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :], two_info, tag_bw], axis=-1)
                    else:
                        fw_input_all = tf.concat([lstm_outputs[:, time_step, :], tag_fw],
                                                 axis=-1)
                        bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :], tag_bw], axis=-1)
                    # forward decoder
                    (fw_cell_output, (fw_c_state, fw_h_state)) = fw_lstm_cell(fw_input_all, (fw_c_state, fw_h_state))
                    tag_fw, _ = self.project(fw_cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_fw = tf.tanh(tag_fw)
                    fw_tag_outputs.append(tag_fw)

                    # backward decoder
                    (bw_cell_output, (bw_c_state, bw_h_state)) = bw_lstm_cell(bw_input_all, (bw_c_state, bw_h_state))
                    tag_bw, _ = self.project(bw_cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_bw = tf.tanh(tag_bw)
                    bw_tag_outputs.append(tag_bw)

                fw_tag_outputs = tf.reshape(tf.transpose(fw_tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                      self.max_seq_len, tag_dim])
                bw_tag_outputs = tf.reshape(tf.transpose(bw_tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                      self.max_seq_len, tag_dim])
                # recover the bw_tag_outputs in order
                bw_tag_outputs = bw_tag_outputs[:, :, ::-1, :]

                tag_final_fw += weight_decay ** layer_id * fw_tag_outputs
                tag_final_list_fw.append(fw_tag_outputs)
                tag_final_bw += weight_decay ** layer_id * bw_tag_outputs
                tag_final_list_bw.append(bw_tag_outputs)
                if self.num_tag_layers > 1:
                    # -----------update event and semantic information for the previous and next setence----------
                    # info aggregation of current sentence, [batch_size, max_doc_len,event_info_h]
                    info_event_sem_current_sent_fw = self.info_agg_layer(tf.concat([fw_tag_outputs, bw_tag_outputs],
                                                                                   axis=-1), reverse_seq=False)
                    # corresponds to the information of previous sentence
                    info_event_sem_pre_sent = tf.concat([info_event_sem_init_sent, info_event_sem_current_sent_fw[:, :-1, :]],
                                                        axis=1)
                    info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                         shape=[self.batch_size * self.max_doc_len, -1])

                    # find valid sentence firstly, and replace with emebedding of unk
                    # if self.reverse_seq:
                    info_event_sem_current_sent_bw = self.info_agg_layer(tf.concat([fw_tag_outputs, bw_tag_outputs],
                                                                                   axis=-1), reverse_seq=self.reverse_seq)
                    # else:
                    #     info_event_sem_current_sent_bw = info_event_sem_current_sent_fw

                    valid_sent_ind = tf.sequence_mask(self.valid_sent_len, maxlen=self.max_doc_len, dtype=tf.float32)
                    valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=2), multiples=[1, 1, self.event_info_h])
                    info_event_sem_current_sent_bw = info_event_sem_current_sent_bw * valid_sent_ind + \
                                                     info_event_sem_mat0 * (1 - valid_sent_ind)

                    # corresponds to the information of previous sentence
                    info_event_sem_next_sent = tf.concat([info_event_sem_current_sent_bw[:, 1:, :], info_event_sem_init_sent],
                                                         axis=1)
                    info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                          shape=[self.batch_size * self.max_doc_len, -1])
        return tag_final_fw, tag_final_bw, tag_final_list_fw, tag_final_list_bw


    def agg_choice_cross_sent_ED(self, words_enc, tag_dim, num_tag_layers, weight_decay, agg_choice="lstm"):
        """
        different choice of aggregation function
        agg_choice: average, lstm, or concat (concat state)
        """
        # decoding layer
        # all layers share the same decoder layer
        # for the first decoder layer, we set c_{i-1} and c_{i+1} with unk_event_semantic
        lstm_outputs = tf.reshape(words_enc, shape=[self.batch_size * self.max_doc_len, self.max_seq_len, -1])
        backward_lstm_outputs = lstm_outputs[:, ::-1, :]

        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                    name="forward_rnn_decoder", reuse=tf.AUTO_REUSE)
        fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=1 - self.dropout_rate)

        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.decode_h, forget_bias=0.0, state_is_tuple=True,
                                                    name="backward_rnn_decoder", reuse=tf.AUTO_REUSE)
        bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=1 - self.dropout_rate)

        # mutli-tagging block
        tag_final_fw = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_bw = tf.zeros(shape=[self.batch_size, self.max_doc_len, self.max_seq_len, tag_dim], dtype=tf.float32)
        tag_final_list_fw = []
        tag_final_list_bw = []

        fw_init_state = fw_lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        bw_init_state = bw_lstm_cell.zero_state(self.batch_size * self.max_doc_len, dtype=tf.float32)
        # event and semantic information of the previous sentence and next sentence sentence
        if agg_choice == "lstm":
            info_event_sem_pre_sent = tf.tile(self.unk_event_semantic,
                                              multiples=[self.batch_size * self.max_doc_len, 1])
            info_event_sem_next_sent = tf.tile(self.unk_event_semantic,
                                               multiples=[self.batch_size * self.max_doc_len, 1])
        else:
            info_event_sem_pre_sent = tf.zeros(shape=[self.batch_size * self.max_doc_len, 1 * tag_dim])
            info_event_sem_next_sent = tf.zeros(shape=[self.batch_size * self.max_doc_len, 1 * tag_dim])

        # event and semantic information of the beginning sentence
        info_event_sem_init_sent = tf.tile(self.unk_event_semantic, multiples=[self.batch_size, 1])
        info_event_sem_init_sent = tf.expand_dims(info_event_sem_init_sent, axis=1)
        info_event_sem_mat0 = tf.tile(tf.expand_dims(self.unk_event_semantic, axis=0),
                                      multiples=[self.batch_size, self.max_doc_len, 1])

        with tf.variable_scope("bidirectional_rnn_decoding_layer", reuse=tf.AUTO_REUSE):
            for layer_id in range(num_tag_layers):
                # initialize for each layer
                fw_c_state, fw_h_state = fw_init_state
                bw_c_state, bw_h_state = bw_init_state
                tag_fw = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                tag_bw = tf.zeros([self.batch_size * self.max_doc_len, tag_dim])
                fw_tag_outputs = []
                bw_tag_outputs = []
                for time_step in range(self.max_seq_len):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    # concat two event information
                    if self.num_tag_layers > 1:
                        if not self.context_info:
                            fw_input_all = tf.concat([lstm_outputs[:, time_step, :], info_event_sem_pre_sent, tag_fw],
                                                     axis=-1)
                            bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :],
                                                      info_event_sem_next_sent, tag_bw], axis=-1)
                        else:
                            two_info = tf.concat([info_event_sem_pre_sent, info_event_sem_next_sent], axis=-1)
                            fw_input_all = tf.concat([lstm_outputs[:, time_step, :], two_info, tag_fw], axis=-1)
                            bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :], two_info, tag_bw],
                                                     axis=-1)
                    else:
                        fw_input_all = tf.concat([lstm_outputs[:, time_step, :], tag_fw],
                                                 axis=-1)
                        bw_input_all = tf.concat([backward_lstm_outputs[:, time_step, :], tag_bw], axis=-1)
                    # forward decoder
                    (fw_cell_output, (fw_c_state, fw_h_state)) = fw_lstm_cell(fw_input_all, (fw_c_state, fw_h_state))
                    tag_fw, _ = self.project(fw_cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_fw = tf.tanh(tag_fw)
                    fw_tag_outputs.append(tag_fw)

                    # backward decoder
                    (bw_cell_output, (bw_c_state, bw_h_state)) = bw_lstm_cell(bw_input_all, (bw_c_state, bw_h_state))
                    tag_bw, _ = self.project(bw_cell_output, tag_dim)
                    if self.event_vector_trans:
                        tag_bw = tf.tanh(tag_bw)
                    bw_tag_outputs.append(tag_bw)

                fw_tag_outputs = tf.reshape(tf.transpose(fw_tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                      self.max_seq_len, tag_dim])
                bw_tag_outputs = tf.reshape(tf.transpose(bw_tag_outputs, [1, 0, 2]), [self.batch_size, self.max_doc_len,
                                                                                      self.max_seq_len, tag_dim])
                # recover the bw_tag_outputs in order
                bw_tag_outputs = bw_tag_outputs[:, :, ::-1, :]

                tag_final_fw += weight_decay ** layer_id * fw_tag_outputs
                tag_final_list_fw.append(fw_tag_outputs)
                tag_final_bw += weight_decay ** layer_id * bw_tag_outputs
                tag_final_list_bw.append(bw_tag_outputs)
                if self.num_tag_layers > 1:
                    # -----------update event and semantic information for the previous and next setence----------
                    # info aggregation of current sentence, [batch_size, max_doc_len,event_info_h]
                    if agg_choice == "lstm":
                        info_event_sem_current_sent_fw = self.info_agg_layer(tf.concat([fw_tag_outputs, bw_tag_outputs],
                                                                                       axis=-1), reverse_seq=False)
                        # corresponds to the information of previous sentence
                        info_event_sem_pre_sent = tf.concat(
                            [info_event_sem_init_sent, info_event_sem_current_sent_fw[:, :-1, :]],
                            axis=1)
                        info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                             shape=[self.batch_size * self.max_doc_len, -1])

                        # find valid sentence firstly, and replace with emebedding of unk
                        info_event_sem_current_sent_bw = self.info_agg_layer(
                            tf.concat([fw_tag_outputs, bw_tag_outputs],axis=-1),
                            reverse_seq=self.reverse_seq)

                        valid_sent_ind = tf.sequence_mask(self.valid_sent_len, maxlen=self.max_doc_len, dtype=tf.float32)
                        valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=2),
                                                 multiples=[1, 1, self.event_info_h])
                        info_event_sem_current_sent_bw = info_event_sem_current_sent_bw * valid_sent_ind + \
                                                         info_event_sem_mat0 * (1 - valid_sent_ind)

                        # corresponds to the information of previous sentence
                        info_event_sem_next_sent = tf.concat(
                            [info_event_sem_current_sent_bw[:, 1:, :], info_event_sem_init_sent],
                            axis=1)
                        info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                              shape=[self.batch_size * self.max_doc_len, -1])
                    elif agg_choice == "average":
                        # two_outputs = tf.concat([fw_tag_outputs, bw_tag_outputs], axis=-1)
                        two_outputs =  (fw_tag_outputs + bw_tag_outputs) / 2
                        dim_tmp = two_outputs.get_shape().as_list()[-1]

                        valid_sent_ind = tf.sequence_mask(self.valid_words_len, maxlen= self.max_seq_len)
                        valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=3), [1, 1, 1, dim_tmp])
                        avg_vect = tf.reduce_sum(two_outputs * tf.cast(valid_sent_ind, dtype=tf.float32), axis=-2)

                        valid_words_inv = tf.tile(tf.expand_dims(1/self.valid_words_len, axis=2),
                                                  [1, 1, dim_tmp])
                        avg_vect = avg_vect * tf.cast(valid_words_inv, dtype=tf.float32)
                        pad_vect = tf.zeros(shape=[self.batch_size, 1, dim_tmp])

                        info_event_sem_pre_sent = tf.concat([pad_vect, avg_vect[:, :-1, :]], axis=1)
                        info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                             shape=[self.batch_size * self.max_doc_len, -1])
                        info_event_sem_next_sent = tf.concat([avg_vect[:, 1:, :], pad_vect], axis=1)
                        info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                             shape=[self.batch_size * self.max_doc_len, -1])
                    elif agg_choice == "concat":
                        """
                        element-wise sum
                        """
                        # two_outputs = tf.concat([fw_tag_outputs, bw_tag_outputs], axis=-1)
                        two_outputs = (fw_tag_outputs + bw_tag_outputs) / 2
                        dim_tmp = two_outputs.get_shape().as_list()[-1]

                        valid_sent_ind = tf.one_hot(self.valid_words_len, depth=self.max_seq_len)
                        valid_sent_ind = tf.tile(tf.expand_dims(valid_sent_ind, axis=3),[1,1,1,dim_tmp])
                        print("shape of two_outputs:\t",two_outputs.get_shape())
                        print("shape of valid_sent_ind:\t",valid_sent_ind.get_shape())

                        first_vect = two_outputs[:, :, 0, :]
                        last_vect = tf.reduce_sum(two_outputs * valid_sent_ind,axis=-2)
                        print("shape of last_vect:\t",last_vect.get_shape())

                        # sent_vect = tf.concat([first_vect, last_vect], axis=-1)
                        sent_vect = (first_vect + last_vect) / 2
                        pad_vect = tf.zeros(shape=[self.batch_size, 1, sent_vect.get_shape().as_list()[-1]])
                        info_event_sem_pre_sent = tf.concat([pad_vect, sent_vect[:, :-1, :]], axis=1)
                        info_event_sem_pre_sent = tf.reshape(info_event_sem_pre_sent,
                                                             shape=[self.batch_size * self.max_doc_len, -1])
                        info_event_sem_next_sent = tf.concat([sent_vect[:, 1:, :], pad_vect], axis=1)
                        info_event_sem_next_sent = tf.reshape(info_event_sem_next_sent,
                                                             shape=[self.batch_size * self.max_doc_len, -1])
                    else:
                        print("agg_choice is not suppoted!")
        return tag_final_fw, tag_final_bw, tag_final_list_fw, tag_final_list_bw


    def fully_connected_layer(self, tag_vects):
        """
        fully connected layer
        """
        tag_vects = tf.nn.dropout(tag_vects, keep_prob=1 - self.dropout_rate)
        enc_dim = tag_vects.get_shape().as_list()[-1]
        with tf.variable_scope("logits"):
            W = tf.get_variable("W",
                                shape=[enc_dim, self.class_size],
                                dtype=tf.float32,
                                initializer=self.initializer
                                )
            b = tf.get_variable("b",
                                shape=[self.class_size],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer()
                                )
            output = tf.reshape(tag_vects, shape=[-1, enc_dim])
            logits_ed = tf.nn.xw_plus_b(output, W, b)
            logits_ed = tf.reshape(logits_ed, [self.batch_size, self.max_doc_len, self.max_seq_len, self.class_size])
        return logits_ed


    def loss_layer(self, tag_vects):
        """
        define the loss function
        """
        # projection layer
        logits_ed = self.fully_connected_layer(tag_vects)

        # calculate loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_ed, labels=self.input_label_docs)

            # mask invalid batches
            mask_batches_0 = tf.sequence_mask(self.valid_batch, maxlen=self.batch_size)
            valid_len_list = tf.boolean_mask(self.valid_words_len, mask_batches_0)
            mask_batches = tf.tile(tf.expand_dims(tf.expand_dims(mask_batches_0, 1), 2),
                                   multiples=[1, self.max_doc_len, self.max_seq_len])
            # mask invalid sents
            mask_sents = tf.sequence_mask(self.valid_sent_len, maxlen=self.max_doc_len)
            valid_len_list = tf.boolean_mask(valid_len_list, tf.boolean_mask(mask_sents, mask_batches_0))
            mask_sents = tf.tile(tf.expand_dims(mask_sents, axis=2), multiples=[1, 1, self.max_seq_len])

            # mask invalid words
            mask_words = tf.sequence_mask(self.valid_words_len, maxlen=self.max_seq_len)

            valid_ind = tf.cast(mask_batches, tf.float32) * tf.cast(mask_sents, tf.float32) * tf.cast(mask_words,
                                                                                                      tf.float32)
            losses = losses * valid_ind

            # weight the loss of positive events
            ind_id_O = tf.cast(tf.equal(self.input_label_docs, self.id_O), tf.float32)
            losses = losses * ind_id_O + self.positive_weights * losses * (1 - ind_id_O)

            loss = tf.reduce_sum(losses) / tf.reduce_sum(valid_ind)

            mask_all_invalid = tf.cast(valid_ind, dtype=tf.bool)

            label_pred = tf.boolean_mask(tf.cast(tf.argmax(logits_ed, axis=-1), tf.float32), mask_all_invalid)
            label_true = tf.boolean_mask(tf.cast(self.input_label_docs, dtype=tf.float32), mask_all_invalid)

            self.final_words_id = tf.boolean_mask(self.input_docs, mask_all_invalid)

        return loss, label_true, label_pred, valid_len_list


if __name__ == "__main__":
    pass
