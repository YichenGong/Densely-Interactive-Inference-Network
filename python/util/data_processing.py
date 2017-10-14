# -*- coding: utf-8 -*-
import numpy as np
import re
import random
import json
import collections
import numpy as np
import util.parameters as params
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn 
import os
import pickle
import multiprocessing
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

FIXED_PARAMETERS, config = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}

base_path = os.getcwd()
nltk_data_path = base_path + "/../TF/nltk_data"
nltk.data.path.append(nltk_data_path)
stemmer = nltk.SnowballStemmer('english')

tt = nltk.tokenize.treebank.TreebankWordTokenizer()

def load_nli_data(path, snli=False, shuffle = True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in tqdm(f):
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True, shuffle = True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def is_exact_match(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    
    token1_stem = stemmer.stem(token1)

    if token1 == token2:
        return True
    
    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == stemmer.stem(lemma):
                return True
    
    if token1 == "n't" and token2 == "not":
        return True
    elif token1 == "not" and token2 == "n't":
        return True
    elif token1_stem == stemmer.stem(token2):
        return True
    return False

def is_antonyms(token1, token2):
    token1 = token1.lower()
    token2 = token2.lower()
    token1_stem = stemmer.stem(token1)
    antonym_lists_for_token2 = []
    for synsets in wn.synsets(token2):
        for lemma_synsets in [wn.synsets(l) for l in synsets.lemma_names()]:
            for lemma_syn in lemma_synsets:
                for lemma in lemma_syn.lemmas():
                    for antonym in lemma.antonyms():
                        antonym_lists_for_token2.append(antonym.name())
                        # if token1_stem == stemmer.stem(antonym.name()):
                        #     return True 
    antonym_lists_for_token2 = list(set(antonym_lists_for_token2))
    for atnm in antonym_lists_for_token2:
        if token1_stem == stemmer.stem(atnm):
            return True
    return False   


def worker(shared_content, dataset):
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    st = StanfordNERTagger('/scratch/yg1053/Stanford/stanford-ner-2014-08-27/classifiers/english.muc.7class.distsim.crf.ser.gz', '/scratch/yg1053/Stanford/stanford-ner-2014-08-27/stanford-ner.jar',encoding='utf-8')
    # st = StanfordNERTagger('/opt/hdfs/user/yichen.gong/nltk/Stanford/stanford-ner-2014-08-27/classifiers/english.muc.7class.distsim.crf.ser.gz', '/opt/hdfs/user/yichen.gong/nltk/Stanford/stanford-ner-2014-08-27/stanford-ner.jar',encoding='utf-8')
    # st = StanfordNERTagger('/opt/hdfs/user/yichen.gong/nltk/Stanford/stanford-ner-2014-08-27/classifiers/english.conll.4class.distsim.crf.ser.gz', '/opt/hdfs/user/yichen.gong/nltk/Stanford/stanford-ner-2014-08-27/stanford-ner.jar',encoding='utf-8')
    NER_classes = "Location, Person, Organization, Money, Percent, Date, Time".split(", ")

    # NER_classes = "Location, Person, Organization, Misc".split(", ")
    NER_indices = {k.upper(): idx for  idx, k in enumerate(NER_classes)}

    print(NER_indices)
    for example in tqdm(dataset):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])

            s1_NER_tags = st.tag(s1_tokenize)
            s2_NER_tags = st.tag(s2_tokenize)
            # print(s1_NER_tags)
            # print(s2_NER_tags)

            s1_NER_feature = [[0]*7 for i in range(len(s1_tokenize))]
            s2_NER_feature = [[0]*7 for i in range(len(s2_tokenize))]

            for idx, pair in enumerate(s1_NER_tags):
                word, tag = pair 
                if tag == "O":
                    continue
                s1_NER_feature[idx][NER_indices[tag]] = 1

            for idx, pair in enumerate(s2_NER_tags):
                word, tag = pair 
                if tag == "O":
                    continue
                s2_NER_feature[idx][NER_indices[tag]] = 1

            # print(s2_NER_feature)



            # s1_token_exact_match = [0] * len(s1_tokenize)
            # s2_token_exact_match = [0] * len(s2_tokenize)
            # s1_token_antonym = [0] * len(s1_tokenize)
            # s2_token_antonym = [0] * len(s2_tokenize)
            # for i, word in enumerate(s1_tokenize):
            #     matched = False
            #     for j, w2 in enumerate(s2_tokenize):
            #         matched = is_exact_match(word, w2)
            #         if matched:
            #             s1_token_exact_match[i] = 1
            #             s2_token_exact_match[j] = 1
                         
                    
            #         antonymed = is_antonyms(word, w2)
            #         if antonymed:
            #             s1_token_antonym[i] = 1
            #             s2_token_antonym[j] = 1
            
            content = {}

            # content['sentence1_token_exact_match_with_s2'] = s1_token_exact_match
            # content['sentence2_token_exact_match_with_s1'] = s2_token_exact_match
            # content['sentence1_antonym_feature'] = s1_token_antonym
            # content['sentence2_antonym_feature'] = s2_token_antonym
            content['sentence1_NER_feature'] = s1_NER_feature
            content['sentence2_NER_feature'] = s2_NER_feature
            shared_content[example["pairID"]] = content
            # print(shared_content[example["pairID"]])
    # print(shared_content)

def load_shared_content(fh, shared_content):
    for line in fh:
        row = line.rstrip().split("\t")
        key = row[0]
        value = json.loads(row[1])
        shared_content[key] = value

def load_mnli_shared_content():
    shared_file_exist = False
    # shared_path = config.datapath + "/shared_2D_EM.json"
    # shared_path = config.datapath + "/shared_anto.json"
    # shared_path = config.datapath + "/shared_NER.json"
    shared_path = config.datapath + "/shared.jsonl"
    # shared_path = "../shared.json"
    print(shared_path)
    if os.path.isfile(shared_path):
        shared_file_exist = True
    # shared_content = {}
    assert shared_file_exist
    # if not shared_file_exist and config.use_exact_match_feature:
    #     with open(shared_path, 'w') as f:
    #         json.dump(dict(reconvert_shared_content), f)
    # elif config.use_exact_match_feature:
    with open(shared_path) as f:
        shared_content = {}
        load_shared_content(f, shared_content)
        # shared_content = json.load(f)
    return shared_content

def sentences_to_padded_index_sequences(datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    
    

    word_counter = collections.Counter()
    char_counter = collections.Counter()
    # mgr = multiprocessing.Manager()
    # shared_content = mgr.dict()
    # process_num = config.num_process_prepro
    # process_num = 1
    for i, dataset in enumerate(datasets):
        # if not shared_file_exist:
        #     num_per_share = len(dataset) / process_num + 1
        #     jobs = [ multiprocessing.Process(target=worker, args=(shared_content, dataset[i * num_per_share : (i + 1) * num_per_share] )) for i in range(process_num)]
        #     for j in jobs:
        #         j.start()
        #     for j in jobs:
        #         j.join()

        for example in tqdm(dataset):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])

            word_counter.update(s1_tokenize)
            word_counter.update(s2_tokenize)

            for i, word in enumerate(s1_tokenize):
                char_counter.update([c for c in word])
            for word in s2_tokenize:
                char_counter.update([c for c in word])

        # shared_content = {k:v for k, v in shared_content.items()}



    


    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    if config.embedding_replacing_rare_word_with_UNK: 
        vocabulary = [PADDING, "<UNK>"] + vocabulary
    else:
        vocabulary = [PADDING] + vocabulary
    # print(char_counter)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
    char_vocab = set([char for char in char_counter])
    char_vocab = list(char_vocab)
    char_vocab = [PADDING] + char_vocab
    char_indices = dict(zip(char_vocab, range(len(char_vocab))))
    indices_to_char = {v: k for k, v in char_indices.items()}
    

    for i, dataset in enumerate(datasets):
        for example in tqdm(dataset):
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                example[sentence + '_inverse_term_frequency'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.float32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)
                      
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                        itf = 0
                    else:
                        if config.embedding_replacing_rare_word_with_UNK:
                            index = word_indices[token_sequence[i]] if word_counter[token_sequence[i]] >= config.UNK_threshold else word_indices["<UNK>"]
                        else:
                            index = word_indices[token_sequence[i]]
                        itf = 1 / (word_counter[token_sequence[i]] + 1)
                    example[sentence + '_index_sequence'][i] = index
                    
                    example[sentence + '_inverse_term_frequency'][i] = itf
                
                example[sentence + '_char_index'] = np.zeros((FIXED_PARAMETERS["seq_length"], config.char_in_word_size), dtype=np.int32)
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        continue
                    else:
                        chars = [c for c in token_sequence[i]]
                        for j in range(config.char_in_word_size):
                            if j >= (len(chars)):
                                break
                            else:
                                index = char_indices[chars[j]]
                            example[sentence + '_char_index'][i,j] = index 
    

    return indices_to_words, word_indices, char_indices, indices_to_char

def get_subword_list(token):
    token = token.lower()
    token = "<" + token + ">"
    subword_list = []
    for i in [3,4,5,6]: 
        for j in range(len(token) - i + 1):
            subword_list.append(token[j : j + i])
    return subword_list

def load_subword_list(sentences, rand = False): 
    list_of_vectors = [] 
    for sentence in sentences:
        sentence_vector = []
        for i in range(config.seq_length): 
            if i < len(sentence):
                idx = range(len(sentence[i]))
                if rand:
                    random.shuffle(idx) 
                
                token_subword_feature_list = [sentence[i][index] for index in idx][:config.subword_feature_len]
                if len(token_subword_feature_list) < config.subword_feature_len:
                    token_subword_feature_list += [0] * (config.subword_feature_len - len(token_subword_feature_list))
                sentence_vector.append(token_subword_feature_list)
            else:
                sentence_vector.append([0] * config.subword_feature_len)
        list_of_vectors.append(sentence_vector)

    return np.array(list_of_vectors)



def parsing_parse(parse):
    base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
    pos = [pair.split(" ")[0] for pair in base_parse]
    return pos

def parse_to_pos_vector(parse, left_padding_and_cropping_pair = (0,0)): # ONE HOT
    pos = parsing_parse(parse)
    pos_vector = [POS_dict.get(tag,0) for tag in pos]
    left_padding, left_cropping = left_padding_and_cropping_pair
    vector = np.zeros((FIXED_PARAMETERS["seq_length"],len(POS_Tagging)))
    assert left_padding == 0 or left_cropping == 0

    for i in range(FIXED_PARAMETERS["seq_length"]):
        if i < len(pos_vector):
            vector[i + left_padding, pos_vector[i + left_cropping]] = 1
        else:
            break
    return vector

def generate_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos_vectors = []
    for parse in parses:
        pos = parsing_parse(parse)
        pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
        pos_vectors.append(pos_vector)

    return construct_one_hot_feature_tensor(pos_vectors, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))

def generate_quora_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos_vectors = []
    for parse in parses:
        pos = parse.split()
        pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
        pos_vectors.append(pos_vector)

    return construct_one_hot_feature_tensor(pos_vectors, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))





def generate_crop_pad_pairs(sequences):
    seq_len = FIXED_PARAMETERS["seq_length"]
    list_of_pairs = []
    for sequence in sequences:
        left_padding = 0
        left_cropping = 0
        if len(sequence) < seq_len:
            left_padding = int(random.uniform(0,1) * (seq_len - len(sequence)))
        elif len(sequence) > seq_len:
            left_cropping = int(random.uniform(0,1) * (len(sequence) - seq_len))
        list_of_pairs.append((left_padding, left_cropping))
    return list_of_pairs


def fill_feature_vector_with_cropping_or_padding(sequences, left_padding_and_cropping_pairs, dim, column_size=None, dtype=np.int32):
    if dim == 1:
        list_of_vectors = []
        for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
            vec = np.zeros((config.seq_length))
            left_padding, left_cropping = pad_crop_pair
            for i in range(config.seq_length):
                if i + left_padding < config.seq_length and i - left_cropping < len(sequence):
                    vec[i + left_padding] = sequence[i + left_cropping]
                else:
                    break
            list_of_vectors.append(vec)
        return np.array(list_of_vectors, dtype=dtype)
    elif dim == 2:
        assert column_size
        tensor_list = []
        for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
            left_padding, left_cropping = pad_crop_pair
            mtrx = np.zeros((config.seq_length, column_size))
            for row_idx in range(config.seq_length):
                if row_idx + left_padding < config.seq_length and row_idx < len(sequence) + left_cropping:
                    for col_idx, content in enumerate(sequence[row_idx + left_cropping]):
                        mtrx[row_idx + left_padding, col_idx] = content
                else:
                    break
            tensor_list.append(mtrx)
        return np.array(tensor_list, dtype=dtype)
    else:
        raise NotImplementedError

def construct_one_hot_feature_tensor(sequences, left_padding_and_cropping_pairs, dim, column_size=None, dtype=np.int32):
    """
    sequences: [[(idx, val)... ()]...[]]
    left_padding_and_cropping_pairs: [[(0,0)...] ... []]
    """
    tensor_list = []
    for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
        left_padding, left_cropping = pad_crop_pair
        if dim == 1:
            vec = np.zeros((config.seq_length))
            for num in sequence:
                if num + left_padding - left_cropping < config.seq_length and num + left_padding - left_cropping >= 0:
                    vec[num + left_padding - left_cropping] = 1
            tensor_list.append(vec)
        elif dim == 2:
            assert column_size
            mtrx = np.zeros((config.seq_length, column_size))
            for row, col in sequence:
                if row + left_padding - left_cropping < config.seq_length and row + left_padding - left_cropping >= 0 and col < column_size:
                    mtrx[row + left_padding - left_cropping, col] = 1
            tensor_list.append(mtrx)

        else:
            raise NotImplementedError

    return np.array(tensor_list, dtype=dtype)





def generate_manual_sample_minibatch(s1_tokenize, s2_tokenize, word_indices, char_indices):

    nst = StanfordNERTagger('/home/users/yichen.gong/Stanford/stanford-ner-2014-08-27/classifiers/english.muc.7class.distsim.crf.ser.gz', '//home/users/yichen.gong/Stanford/stanford-ner-2014-08-27/stanford-ner.jar',encoding='utf-8')
    pst = StanfordPOSTagger('/home/users/yichen.gong/Stanford/stanford-postagger-2014-08-27/models/english-bidirectional-distsim.tagger', \
                        '/home/users/yichen.gong/Stanford/stanford-postagger-2014-08-27/stanford-postagger.jar')
    
    premise_vectors = np.zeros((1, config.seq_length))
    hypothesis_vectors = np.zeros((1, config.seq_length))
    premise_char_vectors = np.zeros((1, config.seq_length, config.char_in_word_size))
    hypothesis_char_vectors = np.zeros((1, config.seq_length, config.char_in_word_size))
    premise_exact_match = np.zeros((1, config.seq_length))
    hypothesis_exact_match = np.zeros((1, config.seq_length))

    for idx, w1 in enumerate(s1_tokenize):
        premise_vectors[0, idx] = word_indices.get(w1, 0)
        for ci, c in enumerate(w1):
            premise_char_vectors[0, idx, ci] = char_indices.get(c, 0)

        for s2idx, w2 in enumerate(s2_tokenize):
            if is_exact_match(w1, w2):
                premise_exact_match[0, idx] = 1 
                hypothesis_exact_match[0, s2idx] = 1

    for idx, w2 in enumerate(s2_tokenize):
        hypothesis_vectors[0, idx] = word_indices.get(w2, 0)
        for ci, c in enumerate(w2):
            hypothesis_char_vectors[0, idx, ci] = char_indices.get(c, 0)

    premise_pos_vectors = np.zeros((1, config.seq_length, len(POS_dict.keys())))
    hypothesis_pos_vectors = np.zeros((1, config.seq_length, len(POS_dict.keys())))

    s1_pos = pst.tag(s1_tokenize)
    s2_pos = pst.tag(s2_tokenize)
    for idx, pair in enumerate(s1_pos):
        word, tag = pair 
        premise_pos_vectors[0, idx, POS_dict[tag]] = 1 

    for idx, pair in enumerate(s2_pos):
        word, tag = pair 
        hypothesis_pos_vectors[0, idx, POS_dict[tag]] = 1


    # s1_ner = nst.tag(s1_tokenize)
    # s2_ner = nst.tag(s2_tokenize)

    # not used
    labels = np.zeros((1))
    genres = np.zeros((1))
    pairIDs = np.zeros((1))
    premise_inverse_term_frequency = np.zeros((1, config.seq_length, 1), dtype=np.float32)
    hypothesis_inverse_term_frequency = np.zeros((1, config.seq_length, 1), dtype=np.float32)
    premise_antonym_feature = np.zeros((1, config.seq_length))
    hypothesis_antonym_feature = np.zeros((1, config.seq_length))

    premise_NER_feature = np.zeros((1, config.seq_length, 7))
    hypothesis_NER_feature = np.zeros((1, config.seq_length, 7))

    premise_exact_match = np.expand_dims(premise_exact_match, 2)
    hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)
    premise_antonym_feature = np.expand_dims(premise_antonym_feature, 2)
    hypothesis_antonym_feature = np.expand_dims(hypothesis_antonym_feature, 2)


    return premise_vectors, hypothesis_vectors, labels, genres, premise_pos_vectors, \
                hypothesis_pos_vectors, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, hypothesis_inverse_term_frequency, \
                premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, hypothesis_NER_feature



def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb
def loadEmbedding_fully_rand(path, word_indices, divident = 1.0):
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1,m), dtype="float32")
    return emb


def loadEmbedding_rand(path, word_indices, divident = 1.0): # TODO double embedding
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    j = 0
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1,m), dtype="float32")
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                except ValueError:
                    print(s[0])
                    continue

    return emb

def all_lemmas(token):
    t = token.lower()
    lemmas = []
    for synsets in wn.synsets(t):
        for lemma in synsets.lemma_names():
            lemmas.append(lemma)
    return list(set(lemmas))
def loadEmbedding_with_lemma(path, word_indices):
    j = 0
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))
    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1,m), dtype="float32")

    records = np.zeros((n))
    indices_to_words = [""] * n 
    for key, val in word_indices.items():
        indices_to_words[val] = key 
    print("OOV words: {}".format(n - np.sum(records) - 1))  
    print("Loading embedding for first round")
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                    records[word_indices[s[0]]] = 1
                except ValueError:
                    print(s[0])
                    continue

    print("OOV words: {}".format(n - np.sum(records) - 1))
    print("Building OOV lemma sets")
    OOV_word_indices = {}
    for i in range(n):
        if records[i] == 0:
            for lemma in all_lemmas(indices_to_words[i]):
                try:
                    OOV_word_indices[lemma].append(i)
                except:
                    OOV_word_indices[lemma] = [i]

    print("Loading embedding for second round")
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in OOV_word_indices:
                for idx in OOV_word_indices[s[0]]:
                    if records[idx] == 0:
                        try:
                            emb[idx, :] = np.asarray(s[1:])
                            records[idx] = 1
                        except ValueError:
                            print(s[0])
                            continue
    print("OOV words: {}".format(n - np.sum(records) - 1))


    return emb

def save_submission(path, ids, pred_ids):
    assert(ids.shape[0] == pred_ids.shape[0])
    reverse_label_map = {str(value): key for key, value in LABEL_MAP.items()}
    f = open(path, 'w')
    f.write("pairID,gold_label\n")
    for i in range(ids.shape[0]):
        pred = pred_ids[i] if not config.force_multi_classes else pred_ids[i] / config.forced_num_multi_classes
        f.write("{},{}\n".format(str(ids[i]), reverse_label_map[str(pred)]))
    f.close()


