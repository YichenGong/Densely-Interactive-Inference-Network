"""

The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.

Required arguements are,
model_type: which model you wish to train with. Valid model types: cbow, bilstm, and esim.
model_name: the name assigned to the model being trained, this will prefix the name of the logs and checkpoint files.
"""

import argparse

parser = argparse.ArgumentParser()

models = ['attmix_CNN', "DIIN"]
def types(s):
    options = [mod for mod in models if s in models]
    if len(options) == 1:
        return options[0]
    return s

# Valid genres to train on. 
genres = ['travel', 'fiction', 'slate', 'telephone', 'government']
def subtypes(s):
    options = [mod for mod in genres if s in genres]
    if len(options) == 1:
        return options[0]
    return s

pa = parser.add_argument

pa("model_type", choices=models, type=types, help="Give model type.")
pa("model_name", type=str, help="Give model name, this will name logs and checkpoints made. For example cbow, esim_test etc.")

pa("--datapath", type=str, default="../data")
pa("--ckptpath", type=str, default="../logs")
pa("--logpath", type=str, default="../logs")
pa("--tbpath", type=str, default="../logs", help='tensorboard path')

pa("--emb_to_load", type=int, default=None, help="Number of embeddings to load. If None, all embeddings are loaded.")
pa("--learning_rate", type=float, default=0.5, help="Learning rate for model")
pa("--keep_rate", type=float, default=1.0, help="Keep rate for dropout in the model")
pa("--input_keep_rate", type=float, default=0.8, help='keep rate for embedding')
pa("--use_input_dropout", action='store_true', help='use input dropout')
pa("--seq_length", type=int, default=48, help="Max sequence length")
pa("--emb_train", action='store_false', help="Call if you want to make your word embeddings trainable.")

pa("--genre", type=str, help="Which genre to train on")
pa("--alpha", type=float, default=0.15, help="What percentage of SNLI data to use in training")

pa("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")
pa("--preprocess_data_only", action='store_true', help='preprocess_data_only')
pa("--num_process_prepro", type=int, default=24, help='num process prepro')

pa("--logit_func", type=str, default="tri_linear", help='logit function')
pa("--dense_att_logit_func", type=str, default="tri_linear", help='logit function on dense attention')

pa("--self_att_logit_func", type=str, default="tri_linear", help='logit function')
pa("--debug_model", action='store_true', help="debug model")
pa("--batch_size", type=int, default=70, help="batch size") ####
pa("--display_step", type=int, default=50, help='display steps')
pa("--eval_step", type=int, default=1000, help='eval step')
pa("--l2_regularization_ratio", type=float, default=9e-5, help='l2 regularization ratio') ##
pa("--training_completely_on_snli", action='store_true', help='train completely on snli')
pa("--use_lr_decay",action='store_true',help='lr decay')
# pa("--lr_decay_rate", type=float, default=0.99, help='lr decay rate')
pa("--use_label_smoothing", action='store_true', help='label smoothing')
pa("--label_smoothing_ratio", type=float, default=0.05, help='label smoothing ratio')

# Switches
pa("--l2_loss", action='store_false', help='have l2 loss') ##
pa("--gradient_clip_value", default=1.0, type=float, help='gradient clip value')
pa("--show_by_step", action='store_true', help='show by step')




pa("--pos_tagging", action='store_true', help="part of speech tagging enabled") ##

pa("--char_emb_size", type=int, default=8, help="char emb size")
pa("--char_in_word_size", type=int, default=16, help="number of chars in word")
pa("--char_out_size", type=int, default = 100, help="char out size")
pa("--use_char_emb", action='store_true', help="use character level info") ##

pa("--highway", action='store_true', help="highway network switch for fusing pos tagging")
pa("--highway_num_layers", type=int, default=2, help='number of highway network')
pa("--highway_network_output_size", type=int, default=None, help='highway_network_output_size')
pa("--additional_concat_out", action='store_true', help="dynamic concat p and h out")


pa("--self_attention_encoding", action='store_false', help='have self attention encoding instead of biLSTM') ##
pa("--self_att_enc_layers", type=int, default=1, help='num layers self att enc') ##



pa("--self_cross_att_enc", action='store_true', help='self cross attention encoding')
pa("--self_cross_att_enc_layers", type=int, default=1, help='self cross attention layers')

pa("--cross_att_residual_conn", action='store_true', help='cross attention residual connection')

pa("--dense_logit_features_num", type=int, default=20, help='dense logit feature number')

pa("--self_att_wo_residual_conn", action='store_true', help='self att without residual connection')
pa("--self_att_fuse_gate_residual_conn", action='store_false', help='self att fuse gate residual connection') ##
pa("--self_att_fuse_gate_relu_z", action='store_true', help='relu instead of tanh')
pa("--conv_fuse_gate_out_origx_base", action='store_true', help='conv_fuse_gate_out_origx_base')
pa("--conv_fuse_gate_out_newx_base", action='store_true', help='conv_fuse_gate_out_newx_base')
pa("--cross_att_fuse_gate_residual_conn", action='store_true', help='cross att fuse gate residual connection')

pa("--use_adagrad_optimizer", action='store_true',help='use adagrad optimizer')
pa("--user_adadeltaOptimizer", action='store_true', help='use adadelta optimizer') ##
pa("--conv_residual_conn_off", action='store_true', help='no more residual connnection in the deep conv net')
pa("--self_att_complex_linear_map_fuse_gate_residual_conn", action='store_true', help='self att complex linear map fuse gate residual connection')
pa("--p_base_fuse_gate", action='store_true', help='p base fuse gate in complex linear map self att')
pa("--tmp_p_base_fuse_gate", action='store_true', help='tmp based fuse gate in complex linear map self att')
pa("--self_att_highway_out", action='store_true', help='highway out for self att')
pa("--self_att_encoding_with_linear_mapping", action='store_true', help='self att encoding with linear mapping')
pa("--wd", type=float, default=0.0, help='weight decay')
pa("--last_avg_pooling", action='store_true', help='last avg pooling')


pa("--highway_use_tanh", action='store_true', help='highway network use tanh activation')
pa("--conv_use_tanh_act", action='store_true', help='tanh becomes the legendary activation function of convolution')
pa("--two_gate_fuse_gate", action='store_false', help='inside fuse gate we have two f gates') ##

pa("--self_att_head_num", type=int, default=3, help='multi-head num')
pa("--use_dense_att_multi_head_self_att", action='store_true', help='use dense attention version of multi head self att')
pa("--dense_attention_dropout", action='store_false', help='dropout on dense attention features') ##
pa("--out_channel_dims", type=str, default="100")
pa("--filter_heights", type=str, default="5")

pa("--wo_conv_dim_matching_res_conn", action='store_true', help='there is no dimension matching and res connect if dim is not matched')



pa("--super_dense_attention", action='store_false', help='super dense attention') ##


pa("--wo_enc_sharing", action='store_false', help='woencsharing') ##
pa("--diff_penalty_loss_ratio", type=float, default=1e-3, help='diff_penalty_loss_ratio') ##

pa("--dropout_keep_rate_decay", action='store_false', help="dropout_keep_rate_decay") ##
pa("--dropout_decay_step",  type=int, default=10000, help='dropout_decay_step') ##
pa("--dropout_decay_rate",  type=float, default=0.977, help='dropout_decay_rate') ##


pa("--sigmoid_growing_l2loss", action='store_false', help='parameterized_l2loss') ##
pa("--weight_l2loss_step_full_reg", type=int, default=100000, help='weight_l2loss_step_full_reg') ##


pa("--transitioning_conv_blocks", action='store_true', help='transitioning conv blocks')
pa("--use_dense_net", action='store_false', help='use dense net') ##
pa("--dense_net_growth_rate", type=int, default=20, help='dense net growth rate') ##
pa("--first_transition_growth_rate", type=int, default=2, help='first_transition_growth_rate')
pa("--dense_net_layers", type=int, default=8, help='dense net layers') ##
pa("--dense_net_bottleneck_size", type=int, default=500, help='dense net bottleneck size')
pa("--dense_net_transition_rate", type=float, default=0.5, help='dense_net_transition_rate') ##
pa("--dense_net_transition_layer_max_pooling", action='store_false', help='dense net transition layer max pooling') ##
pa("--dense_net_wo_bottleneck", action='store_false', help='dense net without bottleneck') ##
pa("--dense_net_act_before_conv", action='store_true', help='dense_net_act_before_conv')
pa("--dense_net_kernel_size", default=3, help='dense net kernel size')
pa("--rm_first_transition_layer", action='store_true', help='rm_first_transition_layer')
pa("--first_scale_down_layer", action='store_false', help='first_scale_down_layer') ##
pa("--first_scale_down_layer_relu", action='store_true', help='first_scale_down_layer_relu') 
pa("--first_scale_down_kernel", type=int, default=1, help='first_scale_down_kernel') ##


pa("--dense_net_first_scale_down_ratio", type=float, default=0.3, help='dense_net_first_scale_down_ratio') ##



pa("--snli_joint_train_with_mnli", action='store_true', help='snli joint train with mnli')




pa("--embedding_replacing_rare_word_with_UNK", action='store_true', help='embedding_replacing_rare_word_with_UNK')
pa("--UNK_threshold", type=int, default=5, help='UNK threshold')


pa("--debug", action='store_true', help='debug mode')
pa("--use_final_state", action='store_true', help='use final states with of the lstm')
pa("--visualize_dense_attention_logits", action='store_true', help='visualize the attention logits in dense attention')



args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl".format(args.datapath),
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath),
        "test_matched": "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl".format(args.datapath),
        "test_mismatched": "{}/multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl".format(args.datapath),
        "training_snli": "{}/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
        "dev_snli": "{}/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
        "test_snli": "{}/snli_1.0/snli_1.0_test.jsonl".format(args.datapath),
        "embedding_data_path": "{}/glove.840B.300d.txt".format(args.datapath),
        "log_path": "{}/{}".format(args.logpath, args.model_name),
        "ckpt_path":  "{}/{}".format(args.ckptpath, args.model_name),
        "embeddings_to_load": args.emb_to_load,
        "word_embedding_dim": 300,
        "hidden_embedding_dim": 300,
        "seq_length": args.seq_length,
        "keep_rate": args.keep_rate, 
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "emb_train": args.emb_train,
        "alpha": args.alpha,
        "genre": args.genre
    }

    return FIXED_PARAMETERS, args

def train_or_test():
    return args.test

