"""
Training script to train a model on MultiNLI and, optionally, on SNLI data as well.
The "alpha" hyperparamaters set in paramaters.py determines if SNLI data is used in training. If alpha = 0, no SNLI data is used in training. If alpha > 0, then down-sampled SNLI data is used in training. 
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
from tqdm import tqdm
import gzip
import pickle
from util.YF import YFOptimizer

FIXED_PARAMETERS, config = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]

if not os.path.exists(FIXED_PARAMETERS["log_path"]):
    os.makedirs(FIXED_PARAMETERS["log_path"])
if not os.path.exists(config.tbpath):
    os.makedirs(config.tbpath)

if config.test:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_test.log"
else:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################


if config.debug_model:
    val_path = os.path.join(config.datapath, "quora_dp_pair_val.jsonl")
    val_data = load_nli_data(val_path)[:499]
    training_data, test_data = val_data, val_data
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([val_data])
else:
    logger.Log("Loading data Quora Duplicate Sentence Pairs")
    train_path = os.path.join(config.datapath, "quora_dp_pair_train.jsonl") 
    val_path = os.path.join(config.datapath, "quora_dp_pair_val.jsonl")
    test_path = os.path.join(config.datapath, "quora_dp_pair_test.jsonl")
    training_data = load_nli_data(train_path)
    val_data = load_nli_data(val_path)
    test_data = load_nli_data(test_path, shuffle=False)

    logger.Log("Loading embeddings")
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_data, 
        val_data, test_data])

config.char_vocab_size = len(char_indices.keys())
#TODO code check
# have path according to snli ratio 
embedding_dir = os.path.join(config.datapath, "embeddings")
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)
embedding_path = os.path.join(embedding_dir, "quora_emb.pkl.gz")


if os.path.exists(embedding_path):
    f = gzip.open(embedding_path, 'rb')
    loaded_embeddings = pickle.load(f)
    f.close()
else:
    glove_path = "/opt/hdfs/user/yichen.gong/data/mnli_data/glove.840B.300d.txt"
    loaded_embeddings = loadEmbedding_rand(glove_path, word_indices)
    f = gzip.open(embedding_path, 'wb')
    pickle.dump(loaded_embeddings, f)
    f.close()


class modelClassifier:
    def __init__(self):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step = config.display_step
        self.eval_step = config.eval_step
        self.save_step = config.eval_step
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        # self.alpha = FIXED_PARAMETERS["alpha"]
        self.config = config

        


        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(self.config, seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

        self.global_step = self.model.global_step

        if config.use_lr_decay:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, config.lr_decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', self.learning_rate)



        # Perform gradient descent with Adam
        if not config.test:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.total_cost, tvars), 1.0)
            if config.use_adagrad_optimizer:
                opt =  tf.train.AdagradOptimizer(self.learning_rate)
                # self.optimizer =.minimize(self.model.total_cost, global_step=global_step)
            elif config.user_adadeltaOptimizer:
                opt = tf.train.AdadeltaOptimizer(self.learning_rate)
                # self.optimizer = .minimize(self.model.total_cost, global_step=global_step)
            elif config.use_yellow_fin_optimizer:
                opt = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0)
            else:
                # opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                # self.optimizer = .minimize(self.model.total_cost, global_step = global_step)
            # self.gvs = opt.compute_gradients(self.model.total_cost)
            self.optimizer = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            # for grad, var in gvs:
            #     print(var.name)
            #     print(grad.name)
            # capped_gvs = [(None, var) if grad is None else (tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
            # self.optimizer = opt.apply_gradients(capped_gvs, global_step = self.global_step)

            # if config.use_yellow_fin_optimizer:
            #     self.optimizer = YFOptimizer(1.0).minimize(self.model.total_cost, global_step=self.global_step)

        

        # tf things: initialize variables and create placeholder for session
        self.tb_writer = tf.summary.FileWriter(config.tbpath)
        logger.Log("Initializing variables")

        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    """{"sentence1_part_of_speech_tagging": "MD PRP VB NNP NNP IN NNP .", "sentence1_binary_parse": "Can you use Vanilla Visa on Amazon ?", "sentence2_parse": "What are some problems with a Vanilla Visa ?", "sentence2_NER_feature": [], "sentence2_token_exact_match_with_s1": [6, 7, 8], "sentence2_binary_parse": "What are some problems with a Vanilla Visa ?", "pairID": "333035", "sentence2": "What are some problems with a Vanilla Visa?", "sentence1_parse": "Can you use Vanilla Visa on Amazon ?", "sentence1_NER_feature": [[6, 0]], "gold_label": "neutral", "sentence2_part_of_speech_tagging": "WP VBP DT NNS IN DT NNP NNP .", "sentence1_token_exact_match_with_s2": [3, 4, 7], "sentence1": "Can you use Vanilla Visa on Amazon?"}"""

    def get_minibatch(self, dataset, start_index, end_index, training=False):
        indices = range(start_index, end_index)

        genres = [['quora'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        pairIDs = np.array([dataset[i]['pairID'] for i in indices])

        if config.random_crop_or_pad_sentence_by_seqlen and training:
            premise_pad_crop_pair = generate_crop_pad_pairs([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices])
            hypothesis_pad_crop_pair = generate_crop_pad_pairs([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices])
        else:
            premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)
        # print("premise_pad_crop_pair")
        # print(premise_pad_crop_pair)
        premise_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
        hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)

        # print("premise_vectors")
        # print(premise_vectors)
        # TODO 
        premise_pos_vectors = generate_quora_pos_feature_tensor([dataset[i]['sentence1_part_of_speech_tagging'][:] for i in indices], premise_pad_crop_pair)
        hypothesis_pos_vectors = generate_quora_pos_feature_tensor([dataset[i]['sentence2_part_of_speech_tagging'][:] for i in indices], hypothesis_pad_crop_pair)
        # print(premise_pos_vectors

        # print("premise_pos_vectors")
        # print(premise_pos_vectors)
        premise_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=config.char_in_word_size)
        hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=config.char_in_word_size)


        # print("premise char vectors")
        # print(premise_char_vectors)
        premise_exact_match = construct_one_hot_feature_tensor([dataset[i]["sentence1_token_exact_match_with_s2"][:] for i in indices], premise_pad_crop_pair, 1)
        hypothesis_exact_match = construct_one_hot_feature_tensor([dataset[i]["sentence2_token_exact_match_with_s1"][:] for i in indices], hypothesis_pad_crop_pair, 1)
        # print("premise_exact_match")
        # print(premise_exact_match)        
        premise_exact_match = np.expand_dims(premise_exact_match, 2)
        hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)

        premise_inverse_term_frequency = hypothesis_inverse_term_frequency =  np.zeros((len(indices), config.seq_length,1))

        # premise_antonym_feature = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_antonym_feature"][:] for i in range(end_index - start_index)], premise_pad_crop_pair, 1)
        # hypothesis_antonym_feature = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_antonym_feature"][:] for i in range(end_index - start_index)], hypothesis_pad_crop_pair, 1)
        premise_antonym_feature = hypothesis_antonym_feature = premise_inverse_term_frequency
        # print("premise_antonym_feature")
        # print(premise_antonym_feature)

        # premise_antonym_feature = np.expand_dims(premise_antonym_feature, 2)
        # hypothesis_antonym_feature = np.expand_dims(hypothesis_antonym_feature, 2)

        premise_NER_feature = construct_one_hot_feature_tensor([dataset[i]["sentence1_NER_feature"][:] for i in indices], premise_pad_crop_pair, 2, 7)
        hypothesis_NER_feature = construct_one_hot_feature_tensor([dataset[i]["sentence2_NER_feature"][:] for i in indices], hypothesis_pad_crop_pair, 2, 7)
        # print("premise_NER_feature")
        # print(premise_NER_feature)


        return premise_vectors, hypothesis_vectors, labels, genres, premise_pos_vectors, \
                hypothesis_pos_vectors, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, hypothesis_inverse_term_frequency, \
                premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, hypothesis_NER_feature


    def train(self, train_quora, dev_quora):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True   
        self.sess = tf.Session(config=sess_config)
        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0
        self.train_dev_set = False
        


        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.completed = False
                self.best_dev_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_quora, self.batch_size)
                self.best_mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_quora[0:5000], self.batch_size)
                logger.Log("Confusion Matrix on dev-quora\n{}".format(confmx))
                
                logger.Log("Restored best Quora Validation acc: %f\n Restored best Quora train acc: %f" %(self.best_dev_mat, self.best_mtrain_acc))
                

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.

        ### Training cycle
        logger.Log("Training...")
        # logger.Log("Model will use %s percent of SNLI data during training" %(self.alpha * 100))

        while True:
            
            training_data = train_quora 

            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            
            # Boolean stating that training has not been completed, 
            self.completed = False 

            # Loop over all batches in epoch
            for i in range(total_batch):

                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
                hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
                hypothesis_NER_feature = self.get_minibatch(training_data, self.batch_size * i, self.batch_size * (i + 1), True)
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: self.keep_rate,
                                self.model.is_train: True,
                                self.model.premise_pos: minibatch_pre_pos,
                                self.model.hypothesis_pos: minibatch_hyp_pos,
                                self.model.premise_char:premise_char_vectors,
                                self.model.hypothesis_char:hypothesis_char_vectors,
                                self.model.premise_exact_match:premise_exact_match,
                                self.model.hypothesis_exact_match: hypothesis_exact_match}

                if self.step % self.display_step == 0:
                    if config.print_gradient:
                        grads = []
                        varss = []
                        for grad , var in self.gvs:
                            if grad is not None:
                                grads.append(grad)
                                varss.append(var)
                        gradients = self.sess.run(grads, feed_dict)
                        for i, grad in enumerate(grads):
                            logger.Log("Gradient for {}".format(varss[i].name))
                            logger.Log(gradients[i])
                    
                    _, c, summary = self.sess.run([self.optimizer, self.model.total_cost, self.model.summary], feed_dict)
                    self.tb_writer.add_summary(summary, self.step)
                    logger.Log("Step: {} completed".format(self.step))
                else:
                    _, c = self.sess.run([self.optimizer, self.model.total_cost], feed_dict)






                if self.step % self.eval_step == 0:
                    if config.print_variables:
                        varss = []
                        for grad , var in self.gvs:
                            varss.append(var)
                        variable_values = self.sess.run(varss[2:], feed_dict)
                        for i, grad in enumerate(varss[2:]):
                            logger.Log("variable value for {}".format(varss[2:][i].name))
                            logger.Log(variable_values[i])

                    dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_quora, self.batch_size)
                    
                    logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))

                    # dev_acc_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                    mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_quora[0:5000], self.batch_size)
                    

                    logger.Log("Step: %i\t Quora Val acc: %f\t Quora train acc: %f" %(self.step, dev_acc_mat, mtrain_acc))
                    logger.Log("Step: %i\t Quora Val cost: %f\t Quora train cost: %f" %(self.step, dev_cost_mat, mtrain_cost))

                if self.step % self.save_step == 0:
                    self.saver.save(self.sess, ckpt_file)
                    
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.02:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_mtrain_acc = mtrain_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(self.best_dev_mat))



                if self.best_dev_mat > 0.88:
                    self.eval_step = 200 
                    self.save_step = 200



                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            
            if (progress < 0.1) or (self.step > self.best_step + 30000):
                logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
                
                self.completed = True
                break

    def classify(self, examples):
        # This classifies a list of examples
        if (test == True) or (self.completed == True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        genres = []
        costs = 0
        for i in tqdm(range(total_batch + 1)):
            if i != total_batch:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
                hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
                hypothesis_NER_feature  = self.get_minibatch(
                    examples, self.batch_size * i, self.batch_size * (i + 1))
            else:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
                hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
                hypothesis_NER_feature  = self.get_minibatch(
                    examples, self.batch_size * i, len(examples))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0,
                                self.model.is_train: False,
                                self.model.premise_pos: minibatch_pre_pos,
                                self.model.hypothesis_pos: minibatch_hyp_pos,
                                self.model.premise_char:premise_char_vectors,
                                self.model.hypothesis_char:hypothesis_char_vectors,
                                self.model.premise_exact_match:premise_exact_match,
                                self.model.hypothesis_exact_match: hypothesis_exact_match}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            costs += cost
            logits = np.vstack([logits, logit])

        if test == True:
            logger.Log("Generating Classification error analysis script")
            correct_file = open(os.path.join(FIXED_PARAMETERS["log_path"], "correctly_classified_pairs.txt"), 'w')
            wrong_file = open(os.path.join(FIXED_PARAMETERS["log_path"], "wrongly_classified_pairs.txt"), 'w')

            pred = np.argmax(logits[1:], axis=1)
            LABEL = ["entailment", "neutral", "contradiction"]
            for i in tqdm(range(pred.shape[0])):
                if pred[i] == examples[i]["label"]:
                    fh = correct_file
                else:
                    fh = wrong_file
                fh.write("S1: {}\n".format(examples[i]["sentence1"].encode('utf-8')))
                fh.write("S2: {}\n".format(examples[i]["sentence2"].encode('utf-8')))
                fh.write("Label:      {}\n".format(examples[i]['gold_label']))
                fh.write("Prediction: {}\n".format(LABEL[pred[i]]))
                fh.write("confidence: \nentailment: {}\nneutral: {}\ncontradiction: {}\n\n".format(logits[1+i, 0], logits[1+i,1], logits[1+i,2]))

            correct_file.close()
            wrong_file.close()
        return genres, np.argmax(logits[1:], axis=1), costs

    def generate_predictions_with_id(self, path, examples):
        if (test == True) or (self.completed == True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        costs = 0
        IDs = np.empty(1)
        for i in tqdm(range(total_batch + 1)):
            if i != total_batch:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
                hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
                hypothesis_NER_feature  = self.get_minibatch(
                    examples, self.batch_size * i, self.batch_size * (i + 1))
            else:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
                hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
                hypothesis_NER_feature  = self.get_minibatch(
                    examples, self.batch_size * i, len(examples))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0,
                                self.model.is_train: False,
                                self.model.premise_pos: minibatch_pre_pos,
                                self.model.hypothesis_pos: minibatch_hyp_pos,
                                self.model.premise_char:premise_char_vectors,
                                self.model.hypothesis_char:hypothesis_char_vectors,
                                self.model.premise_exact_match:premise_exact_match,
                                self.model.hypothesis_exact_match: hypothesis_exact_match}
            logit = self.sess.run(self.model.logits, feed_dict)
            IDs = np.concatenate([IDs, pairIDs])
            logits = np.vstack([logits, logit])
        IDs = IDs[1:]
        logits = np.argmax(logits[1:], axis=1)
        save_submission(path, IDs, logits)






classifier = modelClassifier()

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

# While MultiNLI test-sets aren't released, use dev-sets for testing
# test_matched = dev_matched
# test_mismatched = dev_mismatched

if config.preprocess_data_only:
    pass
elif test == False:
    classifier.train(training_data, val_data)
    logger.Log("Acc on quora dev set: %s" %(evaluate_classifier(classifier.classify, val_data, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on quora test set: %s"%(evaluate_classifier(classifier.classify, test_data, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Generating quora dev pred")
    dev_quora_path = os.path.join(FIXED_PARAMETERS["log_path"], "quora_dev_{}.csv".format(modname))
    classifier.generate_predictions_with_id(dev_quora_path, val_data)
    logger.Log("Generating quora test pred")
    test_quora_path = os.path.join(FIXED_PARAMETERS["log_path"], "quora_test_{}.csv".format(modname))
    classifier.generate_predictions_with_id(test_quora_path, test_data)
else:
    logger.Log("Acc on quora dev set: %s" %(evaluate_classifier(classifier.classify, val_data, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on quora test set: %s"%(evaluate_classifier(classifier.classify, test_data, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Generating quora dev pred")
    dev_quora_path = os.path.join(FIXED_PARAMETERS["log_path"], "quora_dev_{}.csv".format(modname))
    classifier.generate_predictions_with_id(dev_quora_path, val_data)
    logger.Log("Generating quora test pred")
    test_quora_path = os.path.join(FIXED_PARAMETERS["log_path"], "quora_test_{}.csv".format(modname))
    classifier.generate_predictions_with_id(test_quora_path, test_data)
