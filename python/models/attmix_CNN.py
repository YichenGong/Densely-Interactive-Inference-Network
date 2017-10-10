import tensorflow as tf
from util import blocks
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormBasicLSTMCell
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, linear, conv2d, cosine_similarity, variable_summaries, dense_logits, fuse_gate
from my.tensorflow.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from my.tensorflow import flatten, reconstruct, add_wd, exp_mask
import numpy as np

class MyModel(object):
    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train,  embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None):
        ## Define hyperparameters
        # tf.reset_default_graph()
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.LSTM_dim = config.LSTM_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len
        # self.config = config

        ## Define the placeholders
        if config.train_babi:
            self.premise_x = tf.placeholder(tf.int32, [None, self.context_seq_len], name='premise')
            self.hypothesis_x = tf.placeholder(tf.int32, [None, self.query_seq_len], name='hypothesis')
        elif config.subword_random_init_embedding:
            self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length, config.subword_feature_len], name='premise')
            self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length, config.subword_feature_len], name='hypothesis')
        else:
            self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='premise')
            self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='hypothesis')
        self.premise_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='hypothesis_pos')
        self.premise_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='premise_char')
        self.hypothesis_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='hypothesis_char')
        self.premise_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='premise_exact_match')
        self.hypothesis_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='hypothesis_exact_match')
        self.premise_itf = tf.placeholder(tf.float32, [None, self.sequence_length,1], name='premise_itf')
        self.hypothesis_itf = tf.placeholder(tf.float32, [None, self.sequence_length,1], name='hypothesis_itf')
        self.premise_antonym = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='premise_antonym')
        self.hypothesis_antonym = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='hypothesis_antonym')
        self.premise_NER_feature = tf.placeholder(tf.int32, [None, self.sequence_length, 7], name='premise_ner_feature')
        self.hypothesis_NER_feature = tf.placeholder(tf.int32, [None, self.sequence_length, 7], name='hypothesis_ner_feature')
        self.positional_encoding = tf.placeholder(tf.float32, [self.sequence_length, 300], name='positional_encoding')
        if config.add_tensor_to_tensor_dict:
            self.tensor_dict = {}
        else:
            self.tensor_dict = None

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # print(self.global_step.graph)
        if config.dropout_keep_rate_decay:
            self.dropout_keep_rate = tf.train.exponential_decay(config.keep_rate, self.global_step, config.dropout_decay_step, config.dropout_decay_rate, staircase=False, name='dropout_keep_rate')
            config.keep_rate = self.dropout_keep_rate
            tf.summary.scalar('dropout_keep_rate', self.dropout_keep_rate)

        if config.use_label_smoothing:
            self.y = tf.placeholder(tf.float32, [None, 3], name='label_y')
        else:
            self.y = tf.placeholder(tf.int32, [None], name='label_y')
        self.keep_rate_ph = tf.placeholder(tf.float32, [], name='keep_prob')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        ## Define parameters
        # self.E = tf.Variable(embeddings, trainable=emb_train)
        
        ## Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            if config.use_positional_encoding:
                emb = emb + self.positional_encoding
            if config.emb_no_dropout:
                return emb
                # emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
            else:
                # emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
                emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
            return emb_drop

        # Get lengths of unpadded sentences
        if config.subword_random_init_embedding:
            prem_seq_lengths, prem_mask = blocks.length(tf.reduce_sum(self.premise_x, axis=2))
            hyp_seq_lengths, hyp_mask = blocks.length(tf.reduce_sum(self.hypothesis_x, axis=2))
        else: 
            prem_seq_lengths, prem_mask = blocks.length(self.premise_x)  # mask [N, L , 1]
            hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)
        self.prem_mask = prem_mask
        self.hyp_mask = hyp_mask


        ### Embedding layer ###
        if config.subword_random_init_embedding:
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                self.E = tf.Variable(embeddings, trainable=emb_train)
                premise_in = emb_drop(self.E, self.premise_x)
                hypothesis_in = emb_drop(self.E, self.hypothesis_x)
            with tf.variable_scope("subword_emb_sum"):
                premise_in = tf.reduce_sum(premise_in, axis=2)
                hypothesis_in = tf.reduce_sum(hypothesis_in, axis=2) 
                if config.subword_embedding_batch_norm:
                    premise_in = tf.contrib.layers.batch_norm(premise_in)
                    hypothesis_in = tf.contrib.layers.batch_norm(hypothesis_in)               
        else:
            with tf.variable_scope("emb"):
                if config.train_babi:
                    with tf.variable_scope("emb_var"):
                        self.E = tf.get_variable("embedding", shape=[self.pred_size, self.embedding_dim])
                        premise_in = emb_drop(self.E, self.premise_x)   #P
                        hypothesis_in = emb_drop(self.E, self.hypothesis_x)  #H
                else:
                    with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                        self.E = tf.Variable(embeddings, trainable=emb_train)
                        premise_in = emb_drop(self.E, self.premise_x)   #P
                        hypothesis_in = emb_drop(self.E, self.hypothesis_x)  #H


                    # with tf.variable_scope("char_conv"), tf.device("/gpu:0"):
            if config.use_char_emb:
                with tf.variable_scope("char_emb"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[config.char_vocab_size, config.char_emb_size])
                    with tf.variable_scope("char") as scope:
                        char_pre = tf.nn.embedding_lookup(char_emb_mat, self.premise_char)
                        char_hyp = tf.nn.embedding_lookup(char_emb_mat, self.hypothesis_char)

                        filter_sizes = list(map(int, config.out_channel_dims.split(','))) #[100]
                        heights = list(map(int, config.filter_heights.split(',')))        #[5]
                        assert sum(filter_sizes) == config.char_out_size, (filter_sizes, config.char_out_size)
                        with tf.variable_scope("conv") as scope:
                            conv_pre = multi_conv1d(char_pre, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                            scope.reuse_variables()  
                            conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                            conv_pre = tf.reshape(conv_pre, [-1, self.sequence_length, config.char_out_size])
                            conv_hyp = tf.reshape(conv_hyp, [-1, self.sequence_length, config.char_out_size])
                        if config.char_feature_linear:
                            with tf.variable_scope("char_linear") as scope:
                                conv_d = config.char_out_size
                                conv_pre = linear(conv_pre, conv_d , True, bias_start=0.0, scope="char_linear", \
                                        squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                                scope.reuse_variables()
                                conv_hyp = linear(conv_hyp, conv_d , True, bias_start=0.0, scope="char_linear", \
                                        squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                        elif config.char_feature_highway:
                            with tf.variable_scope("char_highway") as scope:
                                conv_pre = highway_network(conv_pre, 1, True, scope='char_conv', wd=config.wd, is_train=self.is_train)
                                scope.reuse_variables()
                                conv_hyp = highway_network(conv_hyp, 1, True, scope='char_conv', wd=config.wd, is_train=self.is_train)
                    premise_in = tf.concat([premise_in, conv_pre], axis=2)
                    hypothesis_in = tf.concat([hypothesis_in, conv_hyp], axis=2)




        if config.pos_tagging:
            premise_in = tf.concat((premise_in, tf.cast(self.premise_pos, tf.float32)), axis=2)
            hypothesis_in = tf.concat((hypothesis_in, tf.cast(self.hypothesis_pos, tf.float32)), axis=2)

        if config.use_exact_match_feature:
            premise_in = tf.concat([premise_in, tf.cast(self.premise_exact_match, tf.float32)], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_exact_match, tf.float32)], axis=2)

        if config.use_inverse_term_frequency_feature:
            premise_in = tf.concat([premise_in, self.premise_itf], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, self.hypothesis_itf], axis=2)

        if config.use_antonym_feature:
            premise_in = tf.concat([premise_in, tf.cast(self.premise_antonym, tf.float32)], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_antonym, tf.float32)], axis=2)

        if config.use_ner_feature:
            premise_in = tf.concat([premise_in, tf.cast(self.premise_NER_feature, tf.float32)], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_NER_feature, tf.float32)], axis=2)

        if config.raw_features:
            raw_pre = premise_in
            raw_hyp = hypothesis_in
        # highway network

        if config.add_tensor_to_tensor_dict:
            self.tensor_dict["premise_with_features"] = premise_in 
            self.tensor_dict["hypothesis_with_features"] = hypothesis_in

        if config.embedding_fuse_gate:
            with tf.variable_scope("embedding_fuse_gate") as scope:
                premise_in = fuse_gate(config, self.is_train, premise_in, premise_in, scope="embedding_fuse_gate")
                scope.reuse_variables()
                hypothesis_in = fuse_gate(config, self.is_train, hypothesis_in, hypothesis_in, scope="embedding_fuse_gate")


        if config.use_input_dropout:
            premise_in = tf.cond(self.is_train, lambda: tf.nn.dropout(premise_in, config.input_keep_rate), lambda: premise_in)
            hypothesis_in = tf.cond(self.is_train, lambda: tf.nn.dropout(hypothesis_in, config.input_keep_rate), lambda: hypothesis_in)


        if config.highway or config.use_char_emb or config.pos_tagging:
            with tf.variable_scope("highway") as scope:
                premise_in = highway_network(premise_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, output_size=config.highway_network_output_size)
                if config.wo_highway_sharing_but_penalize_diff:
                    hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, scope='highway_network_h', wd=config.wd, is_train=self.is_train, output_size=config.highway_network_output_size)
                else:
                    scope.reuse_variables()
                    hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, output_size=config.highway_network_output_size)
        if config.add_tensor_to_tensor_dict:
            self.tensor_dict["premise_after_highway"] = premise_in 
            self.tensor_dict["hypothesis_after_highway"] = hypothesis_in


        # if config.use_positional_encoding:
        #     positional_enc_shape = premise_in.get_shape().as_list()[1:]
        #     print(positional_enc_shape)
        #     positional_encoding = tf.Variable(tf.random_normal(positional_enc_shape, stddev=0.5), name='positional_encoding')
        #     premise_in = premise_in + positional_encoding
        #     hypothesis_in = hypothesis_in + positional_encoding



        if not config.layer_norm_LSTM:
            cell = BasicLSTMCell(self.LSTM_dim, state_is_tuple=True)
        else:
            cell = LayerNormBasicLSTMCell(self.LSTM_dim)
        d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.keep_rate)

        with tf.variable_scope("prepro") as scope:
            # p bilstm
            # tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)
            if config.self_attention_encoding:
                pre = premise_in
                hyp = hypothesis_in
                for i in range(config.self_att_enc_layers):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        p = self_attention_layer(config, self.is_train, pre, p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i)) # [N, len, dim]
                        if config.wo_enc_sharing:
                            h = self_attention_layer(config, self.is_train, hyp, p_mask=hyp_mask, scope="{}_layer_self_att_enc_h".format(i))
                        else:
                            tf.get_variable_scope().reuse_variables()
                            h = self_attention_layer(config, self.is_train, hyp, p_mask=hyp_mask, scope="{}_layer_self_att_enc".format(i))
                        
                        if config.self_att_mul_feature:
                            p = tf.concat([p, p*pre], axis=2)
                            h = tf.concat([h, h*hyp], axis=2)
                        elif config.self_att_diff_feature:
                            p = tf.concat([p, p - pre], axis=2)
                            h = tf.concat([h, h - hyp], axis=2)
                        elif config.self_att_orig_mul_feature:
                            p = tf.concat([pre, p, p * pre], axis=2)
                            h = tf.concat([hyp, h, h * hyp], axis=2)
                        elif config.self_att_orig_diff_mul_feature:
                            p = tf.concat([pre, p, p * pre, p - pre], axis=2)
                            h = tf.concat([hyp, h, h * hyp, h - hyp], axis=2)
                        elif config.self_att_orig_feature:
                            p = tf.concat([p, pre], axis=2)
                            h = tf.concat([h, hyp], axis=2)

                        if config.self_att_encoding_with_linear_mapping:
                            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                                p = linear_mapping_with_residual_conn(config, self.is_train, p, p_mask=prem_mask, scope="{}_layer_self_att_linear_mapping".format(i))
                                tf.get_variable_scope().reuse_variables()
                                h = linear_mapping_with_residual_conn(config, self.is_train, h, p_mask=hyp_mask, scope="{}_layer_self_att_linear_mapping".format(i))
                        variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                        variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))
                        pre = p
                        hyp = h
            elif config.self_cross_att_enc:
                p = premise_in
                h = hypothesis_in
                for i in range(config.self_cross_att_enc_layers):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        p = self_attention_layer(config, self.is_train, p, p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i))
                        tf.get_variable_scope().reuse_variables()
                        h = self_attention_layer(config, self.is_train, h, p_mask=hyp_mask, scope="{}_layer_self_att_enc".format(i))
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        p1 = cross_attention_layer(config, self.is_train, p, h, p_mask=prem_mask, h_mask=hyp_mask, scope="{}_layer_cross_att_enc".format(i))
                        tf.get_variable_scope().reuse_variables()
                        h1 = cross_attention_layer(config, self.is_train, h, p, p_mask=hyp_mask, h_mask=prem_mask, scope="{}_layer_cross_att_enc".format(i))
                        p = p1 
                        h = h1
            elif config.linear_fuse_gate_encoding:
                p = premise_in
                h = hypothesis_in
                for i in range(config.linear_fuse_gate_layers):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        dim = p.get_shape().as_list()[-1]
                        p1 = linear(p, dim ,True, bias_start=0.0, scope="linear_enc_{}".format(i), squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                        p = fuse_gate(config, self.is_train, p, p1, scope="linear_enc_fuse_gate_{}".format(i))
                        tf.get_variable_scope().reuse_variables()
                        h1 = linear(h, dim ,True, bias_start=0.0, scope="linear_enc_{}".format(i), squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                        h = fuse_gate(config, self.is_train, h, h1, scope="linear_enc_fuse_gate_{}".format(i))


            else:
                # (fw_p, bw_p), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell, cell_bw=d_cell, inputs=premise_in, sequence_length=prem_seq_lengths, dtype=tf.float32, scope='p') # [N, L, d] * 2
                # p = tf.concat([fw_p, bw_p], axis=2)
                # # p = tf.concat(2, [fw_p, bw_p])  #[N, L, 2d]
                # # h bilstm
                
                # scope.reuse_variables()
                # (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell, cell_bw=d_cell, inputs=hypothesis_in, sequence_length=hyp_seq_lengths, dtype=tf.float32, scope='p')
                # h = tf.concat([fw_h, bw_h], axis=2) #[N, L, 2d]
                p = premise_in 
                h = hypothesis_in
        
        if config.add_tensor_to_tensor_dict:
            self.tensor_dict["premise_after_self_attention"] = p  
            self.tensor_dict["hypothesis_after_self_attention"] = h

        if config.use_memory_augmentation:
            with tf.variable_scope("mem_augmt") as scope:
                p = memory_augment_layer(config, p, prem_mask, self.is_train, config.memory_key_and_values_num, name="memory_augmentation_layer")
                scope.reuse_variables()
                h = memory_augment_layer(config, h, hyp_mask, self.is_train, config.memory_key_and_values_num, name="memory_augmentation_layer")

        if config.LSTM_encoding:
            with tf.variable_scope("LSTM_encoding") as scope:
                cell = tf.contrib.rnn.LSTMCell(p.get_shape().as_list()[-1])
                d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.keep_rate)

                (fw_p, bw_p), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell, cell_bw=d_cell, inputs=p, sequence_length=prem_seq_lengths, dtype=tf.float32, scope='p') # [N, L, d] * 2
                p_lstm_enc = tf.concat([fw_p, bw_p], axis=2)
                    # p = tf.concat(2, [fw_p, bw_p])  #[N, L, 2d]
                    # h bilstm
                    
                scope.reuse_variables()
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell, cell_bw=d_cell, inputs=h, sequence_length=hyp_seq_lengths, dtype=tf.float32, scope='p')
                h_lstm_enc = tf.concat([fw_h, bw_h], axis=2) #[N, L, 2d]
                if config.lstm_fuse_gate:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        p = fuse_gate(config, self.is_train, p, p_lstm_enc, scope="lstm_enc_fuse_gate")
                        tf.get_variable_scope().reuse_variables()
                        h = fuse_gate(config, self.is_train, h, h_lstm_enc, scope='lstm_enc_fuse_gate')
                else:
                    p = p_lstm_enc
                    h = h_lstm_enc


        with tf.variable_scope("main") as scope:

            def model_one_side(config, main, support, main_length, support_length, main_mask, support_mask, scope):
                with tf.variable_scope(scope or "model_one_side"):
                    # p0 = attention_layer(config, self.is_train, main, support, p_mask=main_mask, h_mask=support_mask, scope="first_hop_att")
                    if config.add_one_d_feature_to_matrix:
                        main = self.add_one_d_feature(config, main, main_mask, scope='main')
                        support = self.add_one_d_feature(config, support, support_mask, scope='support')



                    bi_att_mx = bi_attention_mx(config, self.is_train, main, support, p_mask=main_mask, h_mask=support_mask) # [N, PL, HL]
                    # bi_att_mx = tf.expand_dims(bi_att_mx, 3)
                    print(bi_att_mx.get_shape().as_list())

                    if config.add_tensor_to_tensor_dict:
                        self.tensor_dict["dense_attention"] = bi_att_mx

                    if config.norm_dense_attention_with_last_dim:
                        bi_att_mx = normalize(bi_att_mx)

                    if config.dense_attention_dropout:
                        bi_att_mx = tf.cond(self.is_train, lambda: tf.nn.dropout(bi_att_mx, config.keep_rate), lambda: bi_att_mx)

                    if config.similarity_matrix_dot:
                        bi_att_mx = tf.expand_dims(tf.reduce_sum(bi_att_mx, axis=3) , axis=3)

                    if config.dense_attention_highway:
                        bi_att_mx = highway_network(bi_att_mx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                    elif config.dense_attention_self_fuse_gate:
                        bi_att_mx = fuse_gate(config, self.is_train, bi_att_mx, bi_att_mx, scope="dense_attention_self_fuse_gate")

                    if config.dense_attention_linear:
                        bi_att_mx = linear(bi_att_mx, bi_att_mx.get_shape().as_list()[-1] ,True, bias_start=0.0, scope="DA_linear", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                    is_train=self.is_train)

                    # if config.cos_similarity_feature:
                    #     bi_att_mx = cos_similarity_feature(bi_att_mx)

                    # if config.dense_attention_shuffle_add:
                    bi_att_mx = add_features(config, bi_att_mx, main_mask, support_mask)

                    if config.dense_attention_layer_norm:
                        bi_att_mx = tf.contrib.layers.layer_norm(bi_att_mx)


                    print("DenseAttentionFinalSize")
                    print(bi_att_mx.get_shape().as_list())

                    if config.use_dense_net:
                        out_final = dense_net(config, bi_att_mx, self.is_train, self.tensor_dict)
                    else: #ResNet
                        conv_filters = [int(item) for item in config.conv_filter_size.split(",")]
                        conv_features = [conv_blocks(config, bi_att_mx, fn, "conv_block_knl_{}".format(fn), self.is_train, self.tensor_dict) for fn in conv_filters]
                        out_final = tf.concat(conv_features, axis=1)


                    
                    #max pooling [N, 3. 3. config.res_conv_3_chan]

                    return out_final


            # Vanilla BiDAF & RR

            if config.use_multi_perspective_matching:
                tmp_p = self.multi_perspective_merge(config, p, h, scope = "p_MPM")
                tmp_h = self.multi_perspective_merge(config, h, p, scope = 'h_MPM')
                p = tmp_p
                h = tmp_h

            if config.cross_alignment:  
                tmp_p = cross_attention_layer(config, self.is_train, p, h, prem_mask, hyp_mask,scope = "p_cross_att") #cross_attention_layer(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, tensor_dict=None)
                tmp_h = cross_attention_layer(config, self.is_train, h, p, hyp_mask, prem_mask,scope = 'h_cross_att')
                p = tmp_p
                h = tmp_h

            if config.raw_features:
                p = raw_pre
                h = raw_hyp

            premise_final = model_one_side(config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask, scope="premise_as_main")
            if config.BiBiDAF:
                scope.reuse_variables()
                hypothesis_final = model_one_side(config, h, p, hyp_seq_lengths, prem_seq_lengths, hyp_mask, prem_mask, scope="premise_as_main")
            
            
                if config.diff_mul_output:
                    diff = tf.subtract(premise_final, hypothesis_final)
                    mul = tf.multiply(premise_final, hypothesis_final)
                    f0 = tf.concat((premise_final, hypothesis_final, diff, mul), axis=1)
                elif config.abs_diff_mul_output:
                    diff = tf.abs(tf.subtract(premise_final, hypothesis_final)) 
                    mul = tf.multiply(premise_final, hypothesis_final)
                    f0 = tf.concat((premise_final, hypothesis_final, diff, mul), axis=1)
            else:
                f0 = premise_final

            
                if config.bilinear_out:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        f0 = tf.nn.relu(linear(f0, self.LSTM_dim ,True, bias_start=0.0, scope="bilinear", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                    is_train=self.is_train))

        if config.encoding_layer_classification_loss:
            p_vec = tf.reduce_max(p, axis=1)
            h_vec = tf.reduce_max(h, axis=1)
            if config.without_conv:
                f0 = tf.concat([p_vec, h_vec, p_vec - h_vec, p_vec * h_vec], axis=1) 
            else:
                f0 = tf.concat([f0, p_vec, h_vec, p_vec - h_vec, p_vec * h_vec], axis=1)            



        # Get prediction
        # self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        if config.max_out_logit:
            logits = []
            for k in range(config.max_out_logit_num):
                lgt = linear(f0, self.pred_size ,True, bias_start=0.0, scope="max_out_logit_{}".format(k), squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                is_train=self.is_train)
                logits.append(lgt)
            logtis_aug = [tf.expand_dims(tensor, axis=2) for tensor in logits]
            self.logits = tf.reduce_max(tf.concat(logtis_aug, axis=2), axis=2)
        elif config.squared_out_logit:
            self.logits = linear(f0, self.pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                is_train=self.is_train)
            self.logits = self.logits * self.logits
        else:
            self.logits = linear(f0, self.pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                is_train=self.is_train)

        tf.summary.histogram('logit_histogram', self.logits)

        # Define the cost function
        if config.use_label_smoothing:
            # label_smoothing_ratio = tf.constant(config.label_smoothing_ratio / 3, dtype='float', shape=[], name='label_smoothing_ratio')
            sm_lgt = tf.nn.softmax(self.logits)
            self.total_cost = - tf.reduce_mean(tf.reduce_sum(self.y * tf.log(sm_lgt) + (1 - self.y) * (tf.log( 1 - sm_lgt)), axis=1))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.arg_max(self.y,dimension=1)), tf.float32))
        else:
            self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
        tf.summary.scalar('acc', self.acc)


        if config.use_encoding_layer_classification_loss:
            p_vec = tf.reduce_max(p, axis=1)
            h_vec = tf.reduce_max(h, axis=1)
            cat = tf.concat([p_vec, h_vec, p_vec - h_vec, p_vec * h_vec], axis=1)
            enc_loss_ratio = tf.constant(config.enc_loss_ratio, dtype='float', shape=[], name="encoding_loss_ratio")
            enc_logits = linear(cat, 3 ,True, bias_start=0.0, scope="enc_logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
            self.total_cost += tf.reduce_mean(tf.reductf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=enc_logits)) * enc_loss_ratio
            

        tf.summary.scalar('loss', self.total_cost)

        # calculate acc 
        
        # L2 Loss
        if config.l2_loss:
            if config.sigmoid_growing_l2loss:
                weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") and not tensor.name.endswith("weighted_sum/weights:0")])
                full_l2_step = tf.constant(config.weight_l2loss_step_full_reg , dtype=tf.int32, shape=[], name='full_l2reg_step')
                full_l2_ratio = tf.constant(config.l2_regularization_ratio , dtype='float', shape=[], name='l2_regularization_ratio')
                l2loss_ratio = tf.sigmoid( tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2,tf.float32)) * full_l2_ratio
                tf.summary.scalar('l2loss_ratio', l2loss_ratio)
                l2loss = weights_added * l2loss_ratio
            else:
                l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0")]) * tf.constant(config.l2_regularization_ratio , dtype='float', shape=[], name='l2_regularization_ratio')
            tf.summary.scalar('l2loss', l2loss)
            self.total_cost += l2loss

        if config.wo_enc_sharing or config.wo_highway_sharing_but_penalize_diff and not config.raw_features:
            diffs = []
            for i in range(config.self_att_enc_layers):
                for tensor in tf.trainable_variables():
                    if config.wo_enc_sharing:
                        if tensor.name == "prepro/{}_layer_self_att_enc/self_attention/h_logits/first/weights:0".format(i):
                            l_lg = tensor 
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_attention/h_logits/first/weights:0".format(i):
                            r_lg = tensor 
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_1/weights:0".format(i):    
                            l_fg_lhs_1 = tensor 
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_1/weights:0".format(i):
                            r_fg_lhs_1= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_1/weights:0".format(i):
                            l_fg_rhs_1= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_1/weights:0".format(i):
                            r_fg_rhs_1= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_2/weights:0".format(i):
                            l_fg_lhs_2= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_2/weights:0".format(i):
                            r_fg_lhs_2= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_2/weights:0".format(i):
                            l_fg_rhs_2= tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_2/weights:0".format(i):
                            r_fg_rhs_2= tensor

                        if config.two_gate_fuse_gate:
                            if tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_3/weights:0".format(i):    
                                l_fg_lhs_3 = tensor 
                            elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_3/weights:0".format(i):
                                r_fg_lhs_3 = tensor
                            elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_3/weights:0".format(i):
                                l_fg_rhs_3 = tensor
                            elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_3/weights:0".format(i):
                                r_fg_rhs_3 = tensor
                if config.wo_enc_sharing:
                    diffs += [l_lg - r_lg, l_fg_lhs_1 - r_fg_lhs_1, l_fg_rhs_1 - r_fg_rhs_1, l_fg_lhs_2 - r_fg_lhs_2, l_fg_rhs_2 - r_fg_rhs_2]
                if config.two_gate_fuse_gate:
                    diffs += [l_fg_lhs_3 - r_fg_lhs_3, l_fg_rhs_3 - r_fg_rhs_3]
            for tensor in tf.trainable_variables():
                if config.wo_highway_sharing_but_penalize_diff:
                    if tensor.name == "highway/highway_network/layer_0/trans/weights:0":
                        l_hw_0_trans = tensor     
                    elif tensor.name == "highway/highway_network_h/layer_0/trans/weights:0":
                        r_hw_0_trans = tensor     
                    elif tensor.name == "highway/highway_network/layer_0/gate/weights:0":
                        l_hw_0_gate = tensor     
                    elif tensor.name == "highway/highway_network_h/layer_0/gate/weights:0":
                        r_hw_0_gate = tensor     
                    elif tensor.name == "highway/highway_network/layer_1/trans/weights:0":
                        l_hw_1_trans = tensor     
                    elif tensor.name == "highway/highway_network_h/layer_1/trans/weights:0":
                        r_hw_1_trans = tensor     
                    elif tensor.name == "highway/highway_network/layer_1/gate/weights:0":
                        l_hw_1_gate = tensor     
                    elif tensor.name == "highway/highway_network_h/layer_1/gate/weights:0":
                        r_hw_1_gate = tensor     
            
            
                
            if config.wo_highway_sharing_but_penalize_diff:
                diffs += [l_hw_0_gate - r_hw_0_gate, l_hw_0_trans - r_hw_0_trans, l_hw_1_trans - r_hw_1_trans, l_hw_1_gate - r_hw_1_gate]
            
            if config.sigmoid_growing_l2_diff_loss:
                weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in diffs])
                full_l2_step = tf.constant(config.diff_l2_penalty_full_step , dtype=tf.int32, shape=[], name='full_l2reg_step')
                diff_l2_ratio = tf.constant(config.diff_penalty_loss_ratio , dtype='float', shape=[], name='diff_penalty_loss_ratio')
                diff_l2loss_ratio = tf.sigmoid(tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2,tf.float32)) * diff_l2_ratio
                tf.summary.scalar('diff_l2loss_ratio', diff_l2loss_ratio)
                diff_loss = weights_added * diff_l2loss_ratio
            else:
                diff_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in diffs]) * tf.constant(config.diff_penalty_loss_ratio , dtype='float', shape=[], name='diff_penalty_loss_ratio')
            tf.summary.scalar('diff_penalty_loss', diff_loss)
            self.total_cost += diff_loss

        if config.similarity_penalty_loss:
        # losses = tf.map_fn(lambda x: tf.cond(x[0], lambda: x[1], lambda: 1/(x[1]+0.001)) , (self.y, diff_rel), dtype="float")
            p_vec = tf.reduce_max(p, axis=1)
            h_vec = tf.reduce_max(h, axis=1)
            cos_sim = cosine_similarity(p_vec, h_vec)
            entailment_switch = tf.equal(self.y, tf.constant(0, dtype=tf.int32))
            neutral_switch = tf.equal(self.y, tf.constant(1, dtype=tf.int32))
            contradiction_switch = tf.equal(self.y, tf.constant(2, dtype=tf.int32))

            entailment_loss = tf.map_fn(lambda x: tf.cond(x[0], lambda: 1 / x[1], lambda: tf.constant(0.0, dtype=tf.float32)) , (entailment_switch, cos_sim), dtype="float")
            neutral_loss = tf.map_fn(lambda x: tf.cond(x[0], lambda: tf.abs(x[1]), lambda: tf.constant(0.0, dtype=tf.float32)) , (neutral_switch, cos_sim), dtype="float")
            contradiction_loss = tf.map_fn(lambda x: tf.cond(x[0], lambda: 1 / (-x[1]), lambda: tf.constant(0.0, dtype=tf.float32)) , (contradiction_switch, cos_sim), dtype="float")
            self.total_cost += tf.reduce_mean(tf.add_n([entailment_loss, neutral_loss, contradiction_loss]))


        self.summary = tf.summary.merge_all()

        total_parameters = 0
        for v in tf.global_variables():
            if not v.name.endswith("weights:0") and not v.name.endswith("biases:0"):
                continue
            print(v.name)
            # print(type(v.name))
            shape = v.get_shape().as_list()
            param_num = 1
            for dim in shape:
                param_num *= dim 
            print(param_num)
            total_parameters += param_num
        print(total_parameters)

    def add_one_d_feature(self, config, matrix, mask, scope):
        with tf.variable_scope(scope or "add_one_d_feature"):
            features = []
            if config.add_max_feature_to_sentence:
                features.append(tf.reduce_max(matrix, axis=1))
            if config.add_mean_feature_to_sentence:
                features.append(tf.reduce_mean(matrix, axis=1))
            if config.add_linear_weighted_sum_to_sentence:
                wgt = linear(matrix, 1 ,True, bias_start=0.0, scope="weighted_sum", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                wgt = exp_mask(wgt, mask)
                weighted_sum = tf.reduce_sum(tf.nn.softmax(wgt, dim = 1) * matrix ,axis=1)
                # weighted_sum = tf.Print(weighted_sum,)
                features.append(weighted_sum)

            if config.add_some_linear_weighted_sum_to_sentence:
                wgt = linear(matrix, 8 ,True, bias_start=0.0, scope="weighted_sum", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                list_of_logits = tf.unstack(wgt, axis=2)
                for logit in list_of_logits:
                    # print(logit.get_shape().as_list())
                    logit_tmp = tf.expand_dims(logit, axis=2)
                    # print(logit_tmp.get_shape().as_list())
                    wgt_tmp = exp_mask(logit_tmp, mask) 
                    # print(wgt_tmp.get_shape().as_list())
                    weighted_sum = tf.reduce_sum(tf.nn.softmax(wgt_tmp, dim=1) * matrix, axis=1)
                    features.append(weighted_sum)

            if config.only_some_linear_weighted_sum_to_sentence:
                if config.some_linear_weighted_sum_biliear_logit:
                    tmp_weight = tf.nn.relu(linear(matrix, 200 ,True , bias_start=0.0, scope="weighted_sum_1", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train))
                    wgt = linear(tmp_weight, 48 ,False , bias_start=0.0, scope="weighted_sum", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                else:
                    wgt = linear(matrix, 48 ,False , bias_start=0.0, scope="weighted_sum", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=self.is_train)
                list_of_logits = tf.unstack(wgt, axis=2)
                for logit in list_of_logits:
                    # print(logit.get_shape().as_list())
                    logit_tmp = tf.expand_dims(logit, axis=2)
                    # print(logit_tmp.get_shape().as_list())
                    wgt_tmp = exp_mask(logit_tmp, mask) 
                    # print(wgt_tmp.get_shape().as_list())
                    weighted_sum = tf.reduce_sum(tf.nn.softmax(wgt_tmp, dim=1) * matrix, axis=1)
                    features.append(weighted_sum)
                features = [tf.expand_dims(f, axis=1) for f in features]
                return tf.concat(features, axis=1)

            if config.encoding_dim_as_attention_weight:
                list_of_logits = tf.unstack(matrix, axis=2)[:config.num_encoding_dim_as_attention_weight]
                for logit in list_of_logits:
                    # print(logit.get_shape().as_list())
                    logit_tmp = tf.expand_dims(logit, axis=2)
                    # print(logit_tmp.get_shape().as_list())
                    wgt_tmp = exp_mask(logit_tmp, mask) 
                    # print(wgt_tmp.get_shape().as_list())
                    weighted_sum = tf.reduce_sum(tf.nn.softmax(wgt_tmp, dim=1) * matrix, axis=1)
                    features.append(weighted_sum)
                features = [tf.expand_dims(f, axis=1) for f in features]
                return tf.concat(features, axis=1)

            if len(features) == 0:
                return matrix
            else:
                features = [tf.expand_dims(f, axis=1) for f in features]
                ft = tf.concat(features, axis=1)
                return tf.concat([ft, matrix], axis=1)


    def multi_perspective_merge(self, config, lhs, rhs, scope = None):
        with tf.variable_scope(scope or "multi_perspective_merge"):
            features = []

            if config.MPM_max_pool:
                l = tf.reduce_max(lhs, axis=1)
                r = tf.reduce_max(rhs, axis=1)
                features.append(self.multi_perspective_generation(config, l, r, 16, "MPM_max_pool"))



            if len(features) == 0:
                return lhs 
            else:
                ftr = tf.concat(features, axis=1)
                print("{} out shape".format(scope))
                print(ftr.get_shape().as_list())
                return ftr 



    def multi_perspective_generation(self, config, lhs, rhs, perspectives, scope):
        with tf.variable_scope(scope or "multi_perspective_matching"):
            dim = lhs.get_shape().as_list()[-1]
            comm = lhs * rhs # 
            comm_aug = tf.tile(tf.expand_dims(comm, axis=1), [1, perspectives, 1])
            perspect_weight = tf.get_variable("perspect_weight", shape=[perspectives, dim])
            return comm_aug * perspect_weight




def conv_blocks(config, arg, filter_size, name, is_train, tensor_dict=None):
    with tf.variable_scope(name or "conv_blocks"):
        def conv_pooling(res, name):
            with tf.variable_scope(name or "conv_pooling"):
                chan = res.get_shape().as_list()[-1]
                filters = tf.get_variable("filter", shape=[2,2,chan,chan],dtype='float')
                bias = tf.get_variable("bias", shape=[chan], dtype='float')

                return tf.nn.conv2d(res, filters, [1,2,2,1], "VALID", name='conv_pooling') + bias


        if config.use_elu:
            act = tf.nn.elu 
        elif config.conv_use_tanh_act:
            act = tf.tanh
        elif config.use_selu:
            act = selu 
        elif config.use_PRelu:
            act = PRelu
        else:
            act = tf.nn.relu

        if config.conv_layer_norm:
            norm=tf.contrib.layers.layer_norm
        else:
            norm=tf.contrib.layers.batch_norm


        init_dim = arg.get_shape().as_list()[-1]
        if config.transitioning_conv_blocks:
            res = residual(config, arg, init_dim, 336, filter_size, "res_transition_1", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)    
            res = residual(config, arg, 336, 224, filter_size, "res_transition_2", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)    
            res = residual(config, arg, 224, config.res_conv_1_chan, filter_size, "res1", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        else:
            res = residual(config, arg, init_dim, config.res_conv_1_chan, filter_size, "res1", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        print(res.get_shape().as_list())
        # N * 48 * 48 * config.res_conv_1_chan
        res = residual(config, res, config.res_conv_1_chan, config.res_conv_1_chan, filter_size, "res2", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        print(res.get_shape().as_list())


    # N * 48 * 48 * config.res_conv_1_chan
        # if not config.even_smaller_CNN:
        if not config.rm_1_chan1_conv:
            res = residual(config, res, config.res_conv_1_chan, config.res_conv_1_chan, filter_size, "res3", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        #try more poolings (MAX here) [N, 24, 24, config.res_conv_1_chan]
        if config.use_stride2_conv_replace_max_pooling:
            res = conv_pooling(res, "first_conv_pool")
        else:
            res = tf.nn.max_pool(res, [1,2,2,1], [1,2,2,1], "VALID")
        if not config.even_smaller_CNN:
            res = residual(config, res, config.res_conv_1_chan, config.res_conv_1_chan, filter_size, "res4", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        res = residual(config, res, config.res_conv_1_chan, config.res_conv_1_chan, filter_size, "res5", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        # N * 24 * 24 * config.res_conv_2_chan
        res = residual(config, res, config.res_conv_1_chan, config.res_conv_2_chan, filter_size, "res6", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)

        if config.use_stride2_conv_replace_max_pooling:
            res = conv_pooling(res, "second_conv_pool")
        else:
            res = tf.nn.max_pool(res, [1,2,2,1], [1,2,2,1], "VALID")
        res = residual(config, res, config.res_conv_2_chan, config.res_conv_2_chan, filter_size, "res7", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        if not config.even_smaller_CNN:
            res = residual(config, res, config.res_conv_2_chan, config.res_conv_2_chan, filter_size, "res8", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        if config.add_1_chan2_conv:
            res = residual(config, res, config.res_conv_2_chan, config.res_conv_2_chan, filter_size, "res8_1", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        res = residual(config, res, config.res_conv_2_chan, config.res_conv_3_chan, filter_size, "res9", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)


        if config.use_stride2_conv_replace_max_pooling:
            res = conv_pooling(res, "third_conv_pool")
        else:
            res = tf.nn.max_pool(res, [1,2,2,1], [1,2,2,1], "VALID")
        res = residual(config, res, config.res_conv_3_chan, config.res_conv_3_chan, filter_size, "res13", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        # if not config.even_smaller_CNN:
        if not config.rm_1_chan3_conv:
            res = residual(config, res, config.res_conv_3_chan, config.res_conv_3_chan, filter_size, "res14", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        res = residual(config, res, config.res_conv_3_chan, config.res_conv_3_chan, filter_size, "res15", act = act, norm=norm, is_train = is_train, tensor_dict=tensor_dict)
        if config.last_avg_pooling:
            res = tf.nn.avg_pool(res, [1,6,6,1],[1,1,1,1],"VALID")
        elif config.last_avg_max_pooling:
            max_pool = tf.nn.max_pool(res, [1,6,6,1],[1,1,1,1], "VALID")
            avg_pool = tf.nn.avg_pool(res, [1,6,6,1],[1,1,1,1], "VALID")
            res = tf.concat([max_pool, avg_pool], axis=3)
        elif not config.wo_last_max_pool:
            res = tf.nn.max_pool(res, [1,2,2,1], [1,2,2,1], "VALID")

        shape_list = res.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(res, [-1, shape_list[1]*shape_list[2]*shape_list[3]])
        if config.add_tensor_to_tensor_dict:
            tensor_dict['conv_out_before_reshape'] = res 
            tensor_dict['conv_out_after_reshape'] = out_final
        return out_final

def shuffle_add(config, dense_tensor):
    list_of_logits = tf.unstack(dense_tensor, axis=3)
    np.random.shuffle(list_of_logits)
    list_of_new_logits = []
    for i in range(len(list_of_logits) / 2):
        list_of_new_logits.append(list_of_logits[2 * i] + list_of_logits[2 * i + 1])
    # if config.full_shuffle_add:
    #     np.random.shuffle(list_of_logits)
    #     for i in range(len(list_of_logits) / 2):
    #         list_of_new_logits.append(list_of_logits[2 * i] + list_of_logits[2 * i + 1])
    #     list_of_new_logits = [tf.expand_dims(tensor, axis=3) for tensor in list_of_new_logits]
    #     new_logit = tf.concat(list_of_new_logits, axis=3)
    #     return new_logit
    list_of_new_logits = [tf.expand_dims(tensor, axis=3) for tensor in list_of_new_logits]
    new_logit = tf.concat(list_of_new_logits, axis=3)
    # bi_att_mx = tf.concat([dense_tensor, new_logit], axis=3)
    return new_logit

def add_features(config, dense_attention, p_mask, h_mask):
    features = []
    PL = dense_attention.get_shape().as_list()[1]
    HL = dense_attention.get_shape().as_list()[2]
    # p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
    # h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]
    p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
    h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
    # ph_mask = p_mask_aug & h_mask_aug TODO
    ph_mask = None

    # if config.dense_attention_shuffle_add:
    #     features.append(shuffle_add(config, dense_attention, ph_mask))

    if config.dense_attention_max_feature: #including row-wise softmax, column-wise softmax 
        dense_attention_max_feature(config, dense_attention, features, ph_mask)
        # features.append(dense_attention_max_feature(config, dense_attention, ph_mask))

    if config.dense_attention_mean_feature: #including row-wise softmax, column-wise softmax 
        dense_attention_mean_feature(config, dense_attention, features, ph_mask)

    if config.dense_attention_min_feature: #including row-wise softmax, column-wise softmax 
        dense_attention_min_feature(config, dense_attention, features, ph_mask)

    if config.dense_attention_sum_feature: #including row-wise softmax, column-wise softmax 
        dense_attention_sum_feature(config, dense_attention, features, ph_mask)

    features.append(dense_attention)

    new_dense_attention = tf.concat(features, axis=3)

    return new_dense_attention


def dense_attention_max_feature(config, bi_att_mx, collection, ph_mask):
    sum_feature = tf.reduce_max(bi_att_mx, axis=3)
    collection.append(tf.expand_dims(sum_feature, axis=3))
    switch = [False, False]
    if config.dense_attention_max_row_wise_softmax_feature:
        switch[0] = True

    if config.dense_attention_max_column_wise_softmax_feature:
        switch[1] = True
    dense_logits_softmax_features(config, sum_feature, collection, ph_mask, switch, scope='max_features')
    # return tf.expand_dims(sum_feature, axis=3)

def dense_attention_mean_feature(config, bi_att_mx, collection, ph_mask):
    mean_feature = tf.reduce_mean(bi_att_mx, axis=3)
    collection.append(tf.expand_dims(mean_feature, axis=3))
    switch = [False, False]
    if config.dense_attention_mean_row_wise_feature:
        switch[0] = True

    if config.dense_attention_mean_column_wise_feature:
        switch[1] = True
    dense_logits_softmax_features(config, mean_feature, collection, ph_mask, switch, scope='mean_features')


def dense_attention_min_feature(config, bi_att_mx, collection, ph_mask):
    min_feature = tf.reduce_min(bi_att_mx, axis=3)
    collection.append(tf.expand_dims(min_feature, axis=3))
    switch = [False, False]
    if config.dense_attention_min_row_wise_feature:
        switch[0] = True
    if config.dense_attention_min_column_wise_feature:
        switch[1] = True
    dense_logits_softmax_features(config, min_feature, collection, ph_mask, switch, scope='mean_features')

def dense_attention_sum_feature(config, bi_att_mx, collection, ph_mask):
    sum_feature = tf.reduce_sum(bi_att_mx, axis=3)
    collection.append(tf.expand_dims(sum_feature, axis=3))
    switch = [False, False]
    if config.dense_attention_sum_row_wise_feature:
        switch[0] = True
    if config.dense_attention_sum_column_wise_feature:
        switch[1] = True
    dense_logits_softmax_features(config, sum_feature, collection, ph_mask, switch, scope='mean_features')



def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, tensor_dict=None): #[N, L, 2d]
    with tf.variable_scope(scope or "dense_logit_bi_attention"):
        PL = p.get_shape().as_list()[1]
        HL = h.get_shape().as_list()[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        # if p_mask is None:
        #     ph_mask = None
        # else:
        #     p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
        #     h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
        #     ph_mask = p_mask_aug & h_mask_aug
        ph_mask = None

        if config.super_dense_attention:
            h_logits = p_aug * h_aug
        elif config.super_dense_attention_linear:
            h_logits_tmp = linear(p_aug, p_aug.get_shape().as_list()[-1] ,True, bias_start=0.0, scope="super_dense_attention_linear", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            h_logits = h_logits_tmp * h_aug
        elif config.super_super_dense_attention:
            h_logits = tf.concat([p_aug, h_aug, p_aug * h_aug], axis=3)
        else:
            h_logits = dense_logits(config, [p_aug, h_aug], config.dense_logit_features_num, True, wd=config.wd, mask=ph_mask, is_train=is_train, func=config.dense_att_logit_func, scope='h_logits')  # [N, PL, HL]

        return h_logits

def dense_logits_softmax_features(config, dense_logit_feature, collection, ph_mask, switch , scope=None):
    with tf.variable_scope(scope or "dense_logits_softmax_features"):
        # assert p_mask != None 
        # assert h_mask != None 
        # PL = dense_logit.get_shape().as_list()[1]
        # HL = dense_logit.get_shape().as_list()[2]

        # p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
        # h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
        # ph_mask = p_mask_aug & h_mask_aug #[N, PL, HL]

        # ph_mask_d = tf.tile(tf.expand_dims(ph_mask, 3), [1,1,1,config.dense_logit_features_num])
        dense_logit_with_exp_mask = exp_mask(dense_logit_feature, ph_mask) #[N, PL, HL, 20]
        dense_logit_softmax_col = None 
        dense_logit_softmax_row = None 
        dense_logit_with_exp_mask = tf.expand_dims(dense_logit_with_exp_mask, axis=3)

        if switch[0]:
            print("dense logit with exp mask size")
            print(dense_logit_with_exp_mask.get_shape().as_list())
            dense_logit_softmax_row = tf.nn.softmax(dense_logit_with_exp_mask, dim=2, name='softmax_row')

        if switch[1]:
            dense_logit_softmax_col = tf.nn.softmax(dense_logit_with_exp_mask, dim=1, name='softmax_col')

        

        
        mask = tf.expand_dims(tf.cast(ph_mask,tf.float32), axis=3)
        if dense_logit_softmax_row is not None:
            dense_logit_softmax_row = mask * dense_logit_softmax_row
            print("mask shape")
            print(mask.get_shape().as_list())
            print("single layer feature")
            print(dense_logit_softmax_row.get_shape().as_list())
            collection.append(dense_logit_softmax_row)   
        if dense_logit_softmax_col is not None:
            dense_logit_softmax_col = mask * dense_logit_softmax_col
            collection.append(dense_logit_softmax_col)
        
        # return tf.concat([dense_logit, dense_logit_softmax_col, dense_logit_softmax_row], axis=3)



def self_attention(config, is_train, p, p_mask=None, scope=None, tensor_dict=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape().as_list()[1]
        dim = p.get_shape().as_list()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2

        if config.use_dense_att_multi_head_self_att:
            self_dense_logits = dense_logits(config, [p_aug_1, p_aug_2], config.self_att_head_num, True, bias_start=0.0, scope="dense_logits", mask=self_mask, wd=0.0, input_keep_prob=config.keep_rate, is_train=is_train, func=config.dense_att_logit_func)
            list_of_logits = tf.unstack(self_dense_logits, axis=3)
            list_of_self_att = [softsel(p_aug_2, logit) for logit in list_of_logits]
            self_att = tf.concat(list_of_self_att, axis=2)
            print(self_att.get_shape())
            self_att =  linear(self_att, dim ,True, bias_start=0.0, scope="self_att_rescale", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            print(self_att.get_shape())
            return self_att

        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits) 

        if config.use_multi_head_self_att:
            for i in range(1, config.self_att_head_num):
                print(i)
                with tf.variable_scope("self_att_head_{}".format(i)):
                    p_tmp_1 = linear(p, dim ,True, bias_start=0.0, scope="self_att_head_{}_w1".format(i), squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
                    p_tmp_2 = linear(p, dim ,True, bias_start=0.0, scope="self_att_head_{}_w2".format(i), squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
                    p_aug_tmp_1 = tf.tile(tf.expand_dims(p_tmp_1, 2), [1,1,PL,1])
                    p_aug_tmp_2 = tf.tile(tf.expand_dims(p_tmp_2, 1), [1,PL,1,1]) 
                    logits = get_logits([p_aug_tmp_1, p_aug_tmp_2], None, True, wd=config.wd, mask=self_mask, is_train=is_train, func=config.self_att_logit_func, scope='self_att_head_{}_logit'.format(i))
                    self_att_tmp = softsel(p_aug_tmp_2, logits)
                    self_att = tf.concat([self_att, self_att_tmp], axis=2)
            self_att =  linear(self_att, dim ,True, bias_start=0.0, scope="self_att_rescale", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            print(self_att.get_shape())
        return self_att


def self_attention_layer(config, is_train, p, p_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(config, is_train, p, p_mask=p_mask, tensor_dict=tensor_dict)

        print("self_att shape")
        print(self_att.get_shape())
        if config.self_att_wo_residual_conn:
            p0 = self_att 
        elif config.self_att_fuse_gate_residual_conn:
            p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")
        elif config.self_att_linear_map:
            tmp_p = tf.concat([p, self_att], axis=2)
            p0 = linear(tmp_p, p.get_shape().as_list()[-1] ,True, bias_start=0.0, scope="self_att_linear_map", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        elif config.self_att_complex_linear_map_fuse_gate_residual_conn:
            tmp = tf.concat([p, self_att, p * self_att], axis=2)
            tmp_p = linear(tmp, p.get_shape().as_list()[-1] ,True, bias_start=0.0, scope="self_att_linear_map", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            if config.p_base_fuse_gate:
                p0 = fuse_gate(config, is_train, p, tmp_p, scope='self_att_fuse_gate_p_base')
            elif config.tmp_p_base_fuse_gate:
                p0 = fuse_gate(config, is_train, tmp_p, p, scope='self_att_fuse_gate_tmp_p_base')
            else:
                raise Exception()
        elif config.self_att_highway_out:
             with tf.variable_scope("highway") as scope:
                p0 = highway_network(self_att, config.highway_num_layers, True, wd=config.wd, is_train=is_train)
        else:
            p0 = p + self_att

        if config.att_layer_norm:
            p0 = tf.contrib.layers.layer_norm(p0)

        if config.norm_encoding_with_last_dim:
            p0 = normalize(p0)
        # else:
        #     p0 = tf.concat(3, [p, u_a, p * u_a])
        return p0

def linear_mapping_with_residual_conn(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "linear_mapping"):
        dim = p.get_shape().as_list()[-1]

        p1 = tf.nn.relu(linear(p, dim ,True, bias_start=0.0, scope="linear_maping_1", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train))
        p2 = linear(p1, dim ,True, bias_start=0.0, scope="linear_maping_2", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        return p + p2


def bi_attention(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, h_value = None): #[N, L, 2d]
    with tf.variable_scope(scope or "bi_attention"):
        PL = tf.shape(p)[1]
        HL = tf.shape(h)[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if config.key_value_memory_augmentation:
            #h as key
            #h_value as value
            h_value_aug = tf.tile(tf.expand_dims(h_value, 1), [1,PL,1,1])

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug

        if config.key_value_memory_augmentation:
            h_logits = get_logits([p_aug, h_aug], None, True, wd=config.wd, mask=ph_mask,
                              is_train=is_train, func=config.logit_func, scope='h_logits')  # [N, PL, HL]
            h_a = softsel(h_value_aug, h_logits) 
        else:
            h_logits = get_logits([p_aug, h_aug], None, True, wd=config.wd, mask=ph_mask,
                              is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
            h_a = softsel(h_aug, h_logits) 
        p_a = softsel(p, tf.reduce_max(h_logits, 2))  # [N, 2d]
        p_a = tf.tile(tf.expand_dims(p_a, 1), [1, PL, 1]) # 

        return h_a, p_a


def cross_attention_layer(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "cross_attention_layer"):
        PL = tf.shape(p)[1]
        HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        h_a, p_a = bi_attention(config, is_train, p, h, p_mask=p_mask, h_mask=h_mask)

        if config.att_wo_pa:
            p0 = tf.concat([p, h_a, p * h_a], axis=2)
        else:
            p0 = tf.concat([p, h_a, p * h_a, p * p_a], axis=2)
        # else:
        #     p0 = tf.concat(3, [p, u_a, p * u_a])
        p1 = linear(p0, p.get_shape().as_list()[-1] ,True, bias_start=0.0, scope="cross_att_linear_scale", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        if config.cross_att_residual_conn:
            return p + p1
        elif config.cross_att_fuse_gate_residual_conn:
            return fuse_gate(config, is_train, p, p1, scope="cross_att_fuse_gate")
        else:
            return p1


def residual(config, x, in_filter, out_filter, kernel_size, name, padding = "SAME", activate_before_residual=False, act = tf.nn.relu, norm=tf.contrib.layers.batch_norm, is_train = True, tensor_dict = None):
    # add condition with batch norm
    convolution2d = tf.contrib.layers.convolution2d
    if config.use_inception_structure:
        with tf.variable_scope(name or "inception_CNN"):
            pass
    else:
        with tf.variable_scope(name or "residual_CNN"):
            if activate_before_residual:
                with tf.variable_scope("shared_activation"):
                    if config.CNN_normalize:
                        x = norm(x) 
                    x = act(x)          
                    orig_x = x 
            else:
                with tf.variable_scope("residual_only_activation"):
                    orig_x = x 
                    if config.CNN_normalize:
                        x = norm(x) 
                    x = act(x)
            if config.residual_block_1_3_1:
                with tf.variable_scope("sub1"):
                    if config.CNN_normalize:
                        x = convolution2d(x, out_filter, 1, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    else:
                        x = convolution2d(x, out_filter, 1, padding=padding,  activation_fn=act)
                with tf.variable_scope("sub2"):
                    if config.CNN_normalize:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    else:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding,  activation_fn=act)
                with tf.variable_scope("sub3"):
                    if config.CNN_normalize:
                        x = convolution2d(x, out_filter, 1, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    else:
                        x = convolution2d(x, out_filter, 1, padding=padding,  activation_fn=act)
            elif config.residual_block_dilation:
                with tf.variable_scope("sub1"):
                    if config.residual_block_pre_regular_conv:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding,  activation_fn=act)
                    else:
                        filters = tf.get_variable("weights", shape=[kernel_size,kernel_size,in_filter,out_filter],dtype='float', trainable=True)
                        bias = tf.get_variable("biases", shape=[out_filter], dtype='float')
                        x = tf.nn.atrous_conv2d(x, filters, rate=2, padding=padding) + bias 
                        x = act(x)
                with tf.variable_scope("sub2"):
                    if config.residual_block_post_regular_conv:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding,  activation_fn=act)
                    else:
                        filters = tf.get_variable("weights", shape=[kernel_size,kernel_size,out_filter,out_filter],dtype='float', trainable=True)
                        bias = tf.get_variable("biases", shape=[out_filter], dtype='float')
                        x = tf.nn.atrous_conv2d(x, filters, rate=2, padding=padding) + bias 
                        x = act(x)
            # elif config.residual_2_3_receptive_field:
            #     with tf.variable_scope("sub1"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 2, 2, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2], axis=3)
            #     with tf.variable_scope("sub2"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 2, 2, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2], axis=3)
            # elif config.residual_3_5_receptive_field:
            #     with tf.variable_scope("sub1"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 5, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2], axis=3)
            #     with tf.variable_scope("sub2"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 5, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2], axis=3)
            # elif config.residual_2_3_5_receptive_field:
            #     with tf.variable_scope("sub1"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 4, 2, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub3"):
            #             x3 = convolution2d(x, out_filter / 4, 5, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2, x3], axis=3)
            #     with tf.variable_scope("sub2"):
            #         with tf.variable_scope("sub1"):
            #             x1 = convolution2d(x, out_filter / 4, 2, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub2"):
            #             x2 = convolution2d(x, out_filter / 2, 3, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         with tf.variable_scope("sub3"):
            #             x3 = convolution2d(x, out_filter / 4, 5, padding=padding, normalizer_fn=norm, activation_fn=act) 
            #         x = tf.concat([x1, x2, x3], axis=3)
            else:
                with tf.variable_scope("sub1"):
                    if config.CNN_normalize:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    else:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding,  activation_fn=act)
                    if config.add_tensor_to_tensor_dict:
                        tensor_dict["{}_sub1".format(name)] = x

                if config.conv_inter_dropout:
                    x = tf.cond(is_train, lambda: tf.nn.dropout(x, config.keep_rate), lambda: x)

                with tf.variable_scope("sub2"):
                    if config.CNN_normalize:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    elif config.CNN_layer_2_wo_act:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding, activation_fn = None)
                    else:
                        x = convolution2d(x, out_filter, kernel_size, padding=padding, activation_fn=act)
                    if config.add_tensor_to_tensor_dict:
                        tensor_dict["{}_sub2".format(name)] = x
                if config.conv_end_dropout:
                    x = tf.cond(is_train, lambda: tf.nn.dropout(x, config.keep_rate), lambda: x)                    


            # if config.visualize_dense_attention_logits:
            list_of_conv_features = tf.unstack(x, axis=3)
            # for i in range(len(list_of_logits)):
            tf.summary.image("conv_feature", tf.expand_dims(list_of_conv_features[0],3), max_outputs = 1)

            with tf.variable_scope("sub_add"):
                if in_filter != out_filter: 
                    if config.linear_mapping_conv_mismatch:
                        orig_x = linear(orig_x, out_filter ,True, bias_start=0.0, scope="linear_mapping_conv_mismatch", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                        is_train=is_train)
                    elif config.CNN_normalize:
                        orig_x = convolution2d(orig_x, out_filter, 1, padding=padding, normalizer_fn=norm, activation_fn=act) 
                    elif config.wo_conv_dim_matching_res_conn:
                        return x
                    elif config.mismatch_half_conv_1_channel_replicate_to_add:
                        orig_x = convolution2d(orig_x, out_filter / 2, 1, padding=padding, activation_fn=act) 
                    elif config.mismatch_conv_without_act_for_origx:
                        orig_x = convolution2d(orig_x, out_filter, 1, padding=padding, activation_fn=None) 
                    else:
                        orig_x = convolution2d(orig_x, out_filter, 1, padding=padding, activation_fn=act) 

                if config.conv_fuse_gate_out_origx_base:
                    x = fuse_gate(config, is_train, orig_x, x, scope='conv_fuse_gate')
                elif config.conv_fuse_gate_out_newx_base:
                    x = fuse_gate(config, is_train, x, orig_x, scope='conv_fuse_gate')
                elif config.conv_residual_conn_off:
                    return x
                elif config.conv_shuffle_add_same_mtrx_concat_as_res_conn:
                    x = shuffle_add(config, x)
                    orig_x = shuffle_add(config, orig_x)
                    x = tf.concat([x,orig_x], axis=3)
                elif config.mismatch_half_conv_1_channel_replicate_to_add and in_filter != out_filter:
                    orig_x = tf.concat([orig_x, orig_x], axis=3)
                    x =  x + orig_x
                else:
                    x += orig_x 
                if config.add_tensor_to_tensor_dict:
                    tensor_dict['{}_sub_add'.format(name)] = x
                return x 


def dense_net(config, denseAttention, is_train, tensor_dict = None):
    with tf.variable_scope("dense_net"):
        if config.rm_first_transition_layer:
            fm = denseAttention
        elif config.first_scale_down_maxout:
            dim = denseAttention.get_shape().as_list()[-1]
            act = tf.nn.relu if config.first_scale_down_layer_relu else None
            fms = []
            for k in range(config.first_scale_down_maxout_num):
                afm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding="SAME", activation_fn = act)
                fms.append(afm)
            fms = [tf.expand_dims(tensor, axis=4) for tensor in fms]
            fm = tf.reduce_max(tf.concat(fms, axis=4), axis=4) 
        elif config.first_scale_down_layer:
            dim = denseAttention.get_shape().as_list()[-1]
            act = tf.nn.relu if config.first_scale_down_layer_relu else None
            fm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding="SAME", activation_fn = act)
            if config.add_tensor_to_tensor_dict:
                tensor_dict["first_scale_down_layer"] = fm
        else:
            fm = dense_net_transition_layer(config, denseAttention, config.first_transition_growth_rate, scope='first_transition_layer', tensor_dict=tensor_dict)
        if config.dense_net_skip_join:
            fm_tmp = fm 

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "first_dense_net_block", tensor_dict=tensor_dict) 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer', tensor_dict=tensor_dict)
        if config.dense_net_skip_join:
            fm, fm_tmp = dense_net_skip_join(fm, fm_tmp) 
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "second_dense_net_block", tensor_dict=tensor_dict) 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer', tensor_dict=tensor_dict)
        if config.dense_net_skip_join:
            fm, fm_tmp = dense_net_skip_join(fm, fm_tmp) 

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "third_dense_net_block", tensor_dict=tensor_dict) 
        if config.replace_last_transition_layer_with_residual_block:
            dim = fm.get_shape().as_list()[-1]
            fm = residual(config, fm, dim, dim, 3, "last_layer_in_dense_block", padding = "SAME", act = tf.nn.relu, is_train = is_train)
            if config.add_max_pool_to_last_residual_block:
                fm = tf.nn.max_pool(fm, [1,2,2,1],[1,2,2,1], "VALID")
        else:
            fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer', tensor_dict=tensor_dict)
        if config.fourth_dense_block:
            fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "fourth_dense_net_block", tensor_dict=tensor_dict)
        if config.dense_net_skip_join:
            fm, fm_tmp = dense_net_skip_join(fm, fm_tmp) 
            # fm_tmp = fm 
        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(fm, [-1, shape_list[1]*shape_list[2]*shape_list[3]])
        return out_final



def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train ,padding="SAME", act=tf.nn.relu, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):
            # if config.dense_net_wo_bottleneck:
            if config.dense_net_act_before_conv:
                if config.BN_on_dense_net_block:
                    ft = tf.contrib.layers.batch_norm(features)
                    ft = act(ft)
                else:
                    ft = act(features)
                ft = conv2d(ft, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=None)
            else:
                if config.dense_net_dilated_CNN and i % config.dense_net_dilated_CNN_layers_jump_step == 0:
                    ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act, rate = (2,2))
                else:
                    ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
                # if config.dual_path_network_on_dense_net:
                #     res = conv2d(features, dim, (kernel_size, kernel_size), padding=padding, activation_fn=act)
                #     list_of_features[0] += res 
            # else:
            #     if config.dense_net_act_before_conv:
            #         bt = act(features)
            #         bt = conv2d(bt, config.dense_net_bottleneck_size, 1, padding=padding, activation_fn=None)
            #         ft = act(bt)
            #         ft = conv2d(bt, growth_rate, kernel_size, padding=padding, activation_fn=None)
            #     else:
            #         bt = conv2d(features, config.dense_net_bottleneck_size, 1, padding=padding, activation_fn=act)
            #         ft = conv2d(bt, growth_rate, kernel_size, padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)
        if config.discard_orig_feature_map_to_save_transition_layer:
            return tf.concat(list_of_features[1:], axis=3)

        if config.norm_dense_block_with_last_dim:
            features = normalize(feature_map)

        if config.add_tensor_to_tensor_dict:
            tensor_dict[scope] = features
        print("dense net block out shape")
        print(features.get_shape().as_list())
        if config.dense_net_block_dropout_at_the_end:
            features = tf.cond(is_train, lambda: tf.nn.dropout(features, config.keep_rate), lambda: features)

        return features 

def dense_net_transition_layer(config, feature_map, transition_rate, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "transition_layer"):
        if config.BN_on_dense_net_transition_layer:
            feature_map = tf.contrib.layers.batch_norm(feature_map)

        if config.transition_layer_pooling_first_then_scale_down:
            feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")

        if config.discard_orig_feature_map_to_save_transition_layer:
            feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")
            return feature_map

        if config.addition_as_transition_scale_down_layer:
            fm_list = [tf.expand_dims(tensor, axis=3) for tensor in tf.unstack(feature_map, axis=3)]
            features_map = tf.concat([fm_list[2 * i] + fm_list[2 * i + 1] for i in range(len(fm_list) / 2)], axis=3)
        else:
            out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
            feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn = None)
        # if config.dense_net_transition_layer_max_pooling:
        if not config.transition_layer_pooling_first_then_scale_down:
            feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")
        # else:
        #     feature_map = tf.nn.avg_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")

        if config.norm_transition_block_with_last_dim:
            feature_map = normalize(feature_map)

        if config.add_tensor_to_tensor_dict:
            tensor_dict[scope] = feature_map
        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map


def dense_net_skip_join(fm, fm_tmp):
    down_sampled_tf_tmp = tf.nn.max_pool(fm_tmp, [1,2,2,1],[1,2,2,1], "VALID")
    fm = tf.concat([fm, down_sampled_tf_tmp], axis=3)
    return fm, fm 



def memory_augment_layer(config, x, x_mask, is_train, memory_size, name=None):
    with tf.variable_scope(name or "memory_augmentation"):
        length = x.get_shape().as_list()[-2]
        dim = x.get_shape().as_list()[-1]
        if config.key_value_memory_augmentation:
            out = x 
            for i in range(config.memory_augment_layers):
                keys = tf.get_variable("memory_keys_{}".format(i), shape=[memory_size, dim])
                values = tf.get_variable("memory_values_{}".format(i), shape=[memory_size, dim])

                mem_mask = tf.ones([memory_size, 1], name='memory_mask')
                mem_mask_aug = tf.expand_dims(mem_mask, 0)
                key_aug = tf.expand_dims(keys, 0)
                value_aug = tf.expand_dims(values, 0)
                attended_x , _ = bi_attention(config, is_train, x, key_aug, p_mask=x_mask, h_mask=mem_mask_aug, scope="attend_x", h_value=value_aug)
                if config.memory_augment_layer_add_out:
                    out = out + attended_x
                else:
                    out = fuse_gate(config, is_train, out, attended_x, scope="fuse_gate_memory")

                tf.summary.image("memory_{}_layer_keys".format(i), tf.expand_dims(tf.expand_dims(keys,2),0), max_outputs = 1)
                tf.summary.image("memory_{}_layer_values".format(i), tf.expand_dims(tf.expand_dims(values,2),0), max_outputs = 1)

                variable_summaries(keys, "memory_keys_{}".format(i))
                variable_summaries(values, "memory_values_{}".format(i))



        else: #attentional memory augmentation
            mem = tf.get_variable("memory_key_and_value", shape=[memory_size, dim])
            mem_mask = tf.ones([memory_size, 1], name='memory_mask')
            mem_aug = tf.expand_dims(mem, 0)
            mem_mask_aug = tf.expand_dims(mem_mask, 0)
            attended_x , _ = bi_attention(config, is_train, x, mem_aug, p_mask=x_mask, h_mask=mem_mask_aug, scope="attend_x")
            if config.memory_augment_layer_add_out:
                out = x + attended_x
            else:
                out = fuse_gate(config, is_train, x, attended_x, scope="fuse_gate_memory")
        return out 


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def PRelu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                            initializer = tf.constant_initializer(0.0),
                            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5 

    return pos + neg

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs
