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


FIXED_PARAMETERS, config = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]

if not os.path.exists(FIXED_PARAMETERS["log_path"]):
    os.makedirs(FIXED_PARAMETERS["log_path"])
if not os.path.exists(config.tbpath):
    os.makedirs(config.tbpath)
    config.tbpath = FIXED_PARAMETERS["log_path"]

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
    # training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_matched, test_mismatched = [],[],[],[],[],[], [], []
    test_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"], shuffle = False)[:499]
    training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_mismatched = test_matched, test_matched,test_matched,test_matched,test_matched,test_matched,test_matched
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([test_matched])
    shared_content = load_mnli_shared_content()
else:

    logger.Log("Loading data SNLI")
    training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
    dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
    test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

    logger.Log("Loading data MNLI")
    training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
    dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
    dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

    test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], shuffle = False)
    test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], shuffle = False)

    shared_content = load_mnli_shared_content()

    logger.Log("Loading embeddings")
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli])

config.char_vocab_size = len(char_indices.keys())


embedding_dir = os.path.join(config.datapath, "embeddings")
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

print("embedding path exist")
print(os.path.exists(embedding_path))
if os.path.exists(embedding_path):
    f = gzip.open(embedding_path, 'rb')
    loaded_embeddings = pickle.load(f)
    f.close()
else:
    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)
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
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.config = config

        


        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(self.config, seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

        self.global_step = self.model.global_step



        # Perform gradient descent with Adam
        if not config.test:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.total_cost, tvars), config.gradient_clip_value)
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            self.optimizer = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step)


        # tf things: initialize variables and create placeholder for session
        self.tb_writer = tf.summary.FileWriter(config.tbpath)
        logger.Log("Initializing variables")

        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()





    def get_minibatch(self, dataset, start_index, end_index, training=False):
        indices = range(start_index, end_index)

        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        pairIDs = np.array([dataset[i]['pairID'] for i in indices])


        premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)


        premise_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
        hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)
        premise_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=config.char_in_word_size)
        hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=config.char_in_word_size)

        premise_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence1_parse'][:] for i in indices], premise_pad_crop_pair)
        hypothesis_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence2_parse'][:] for i in indices], hypothesis_pad_crop_pair)

        premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(indices))], premise_pad_crop_pair, 1)
        hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(indices))], hypothesis_pad_crop_pair, 1)
  
        premise_exact_match = np.expand_dims(premise_exact_match, 2)
        hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)


        return premise_vectors, hypothesis_vectors, labels, genres, premise_pos_vectors, \
                hypothesis_pos_vectors, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match
                


    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):
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
        self.dont_print_unnecessary_info = False
        self.collect_failed_sample = False

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.completed = False
                dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                best_dev_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))
                if self.alpha != 0.:
                    self.best_strain_acc, strain_cost, _  = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f\n Restored best SNLI train acc: %f" %(dev_acc_mat, best_dev_mismat, best_dev_snli,  self.best_mtrain_acc,  self.best_strain_acc))
                else:
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f" %(dev_acc_mat, best_dev_mismat, best_dev_snli, self.best_mtrain_acc))
                if config.training_completely_on_snli:
                    self.best_dev_mat = best_dev_snli
            else:
                self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.
        beta = int(self.alpha * len(train_snli))

        ### Training cycle
        logger.Log("Training...")
        logger.Log("Model will use %s percent of SNLI data during training" %(self.alpha * 100))

        while True:
            if config.training_completely_on_snli:
                training_data = train_snli
                beta = int(self.alpha * len(train_mnli))
                if config.snli_joint_train_with_mnli:
                    training_data = train_snli + random.sample(train_mnli, beta)

            else:
                training_data = train_mnli + random.sample(train_snli, beta)
                
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
                premise_exact_match, hypothesis_exact_match  = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1), True)
                
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
                    _, c, summary, logits = self.sess.run([self.optimizer, self.model.total_cost, self.model.summary, self.model.logits], feed_dict)
                    self.tb_writer.add_summary(summary, self.step)
                    logger.Log("Step: {} completed".format(self.step))
                else:
                    _, c, logits = self.sess.run([self.optimizer, self.model.total_cost, self.model.logits], feed_dict)


                if self.step % self.eval_step == 0:
                    if config.training_completely_on_snli and self.dont_print_unnecessary_info:
                        dev_acc_mat = dev_cost_mat = 1.0
                    else:
                        dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                        logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))
                    
                    if config.training_completely_on_snli:
                            dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                            dev_acc_mismat, dev_cost_mismat = 0,0
                    elif not self.dont_print_unnecessary_info or 100 * (1 - self.best_dev_mat / dev_acc_mat) > 0.04:
                        dev_acc_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                        dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    else:
                        dev_acc_mismat, dev_cost_mismat, dev_acc_snli, dev_cost_snli = 0,0,0,0

                    if self.dont_print_unnecessary_info and config.training_completely_on_snli:
                        mtrain_acc, mtrain_cost, = 0, 0
                    else:
                        mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                    
                    if self.alpha != 0.:
                        if not self.dont_print_unnecessary_info or 100 * (1 - self.best_dev_mat / dev_acc_mat) > 0.04:
                            strain_acc, strain_cost,_ = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                        elif config.training_completely_on_snli:
                            strain_acc, strain_cost,_ = evaluate_classifier(self.classify, train_snli[0:5000], self.batch_size)
                        else:
                            strain_acc, strain_cost = 0, 0
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f\t SNLI train acc: %f" %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc, strain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f\t SNLI train cost: %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost, strain_cost))
                    else:
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f" %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost))

                if self.step % self.save_step == 0:
                    self.saver.save(self.sess, ckpt_file)
                    if config.training_completely_on_snli:
                        dev_acc_mat = dev_acc_snli
                        mtrain_acc = strain_acc
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_mtrain_acc = mtrain_acc
                        if self.alpha != 0.:
                            self.best_strain_acc = strain_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(self.best_dev_mat))

                if self.best_dev_mat > 0.777 and not config.training_completely_on_snli:
                    self.eval_step = 500
                    self.save_step = 500
                    



                if self.best_dev_mat > 0.780 and not config.training_completely_on_snli:
                    self.eval_step = 100
                    self.save_step = 100
                    self.dont_print_unnecessary_info = True 
                    # if config.use_sgd_at_the_end:
                    #     self.optimizer =  tf.train.GradientDescentOptimizer(0.00001).minimize(self.model.total_cost, global_step = self.global_step)


                if self.best_dev_mat > 0.872 and config.training_completely_on_snli:
                    self.eval_step = 500
                    self.save_step = 500
                
                if self.best_dev_mat > 0.878 and config.training_completely_on_snli:
                    self.eval_step = 100
                    self.save_step = 100
                    self.dont_print_unnecessary_info = True 



                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            self.early_stopping_step = 35000
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 


            if (progress < 0.1) or (self.step > self.best_step + self.early_stopping_step):
                logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
                if config.training_completely_on_snli:
                    self.train_dev_set = True

                    # if dev_cost_snli < strain_cost:
                    self.completed = True
                    break
                else:
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
        pred_size = 3 
        logits = np.empty(pred_size)
        genres = []
        costs = 0
        
        for i in tqdm(range(total_batch + 1)):
            if i != total_batch:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match  = self.get_minibatch(
                    examples, self.batch_size * i, self.batch_size * (i + 1))
            else:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match = self.get_minibatch(
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
        pred_size = 3
        logits = np.empty(pred_size)
        costs = 0
        IDs = np.empty(1)
        for i in tqdm(range(total_batch + 1)):
            if i != total_batch:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match = self.get_minibatch(
                    examples, self.batch_size * i, self.batch_size * (i + 1))
            else:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match = self.get_minibatch(
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
        save_submission(path, IDs, logits[1:])






classifier = modelClassifier()

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()


if config.preprocess_data_only:
    pass
elif test == False:
    classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
    logger.Log("Acc on matched multiNLI dev-set: %s" %(evaluate_classifier(classifier.classify, dev_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on mismatched multiNLI dev-set: %s" %(evaluate_classifier(classifier.classify, dev_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on SNLI test-set: %s" %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])



    if config.training_completely_on_snli:
        logger.Log("Generating SNLI dev pred")
        dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_snli_path, dev_snli)

        logger.Log("Generating SNLI test pred")
        test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
        classifier.generate_predictions_with_id(test_snli_path, test_snli)
        
    else:
        logger.Log("Generating dev matched answers.")
        dev_matched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_matched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_matched_path, dev_matched)
        logger.Log("Generating dev mismatched answers.")
        dev_mismatched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_mismatched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_mismatched_path, dev_mismatched)

else:
    if config.training_completely_on_snli:
        logger.Log("Generating SNLI dev pred")
        dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_snli_path, dev_snli)

        logger.Log("Generating SNLI test pred")
        test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
        classifier.generate_predictions_with_id(test_snli_path, test_snli)
        
    else:
        logger.Log("Evaluating on multiNLI matched dev-set")
        matched_multinli_dev_set_eval = evaluate_classifier(classifier.classify, dev_matched, FIXED_PARAMETERS["batch_size"])
        logger.Log("Acc on matched multiNLI dev-set: %s" %(matched_multinli_dev_set_eval[0]))
        logger.Log("Confusion Matrix \n{}".format(matched_multinli_dev_set_eval[2]))

        logger.Log("Generating dev matched answers.")
        dev_matched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_matched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_matched_path, dev_matched)
        logger.Log("Generating dev mismatched answers.")
        dev_mismatched_path = os.path.join(FIXED_PARAMETERS["log_path"], "dev_mismatched_submission_{}.csv".format(modname))
        classifier.generate_predictions_with_id(dev_mismatched_path, dev_mismatched)
