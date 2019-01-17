"""
Preprocess some NLI dataset and word embeddings to be used by the ESIM model.
"""

import os
import torch
import fnmatch
import numpy as np
from fastNLP import DataSet
from fastNLP import Vocabulary
from fastNLP.io.config_io import ConfigSection, ConfigLoader

import fastNLP.core.utils as util


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_premise(ins):
    parentheses_table = str.maketrans({'(': None, ')': None})
    return ins['premise'].translate(parentheses_table).rstrip().split()


def split_hypothesis(ins):
    parentheses_table = str.maketrans({'(': None, ')': None})
    return ins['hypothesis'].translate(parentheses_table).rstrip().split()


def read_data(filepath):
    """
    Read the premises, hypotheses and labels from a file in some NLI
    dataset and return them.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (or MultiNLI) dataset.

    Returns:
        fastNLP.DataSet
    """
    dataset = DataSet.read_csv(filepath, sep='\t')

    fields = list(dataset.get_all_fields())

    for field in fields[3:]:
        dataset.delete_field(field)

    field0 = fields[0]
    field1 = fields[1]
    field2 = fields[2]
    dataset.rename_field(field0, 'label')
    dataset.rename_field(field1, 'premise')
    dataset.rename_field(field2, 'hypothesis')
    dataset.drop(lambda x: x['label'] == '-')

    dataset.apply(split_premise, new_field_name='premise')
    dataset.apply(split_hypothesis, new_field_name='hypothesis')

    dataset.apply(lambda x: ['_BOS_'] + x['premise'] + ['_EOS_'], new_field_name='premise')
    dataset.apply(lambda x: ['_BOS_'] + x['hypothesis'] + ['_EOS_'], new_field_name='hypothesis')

    return dataset


def build_worddict(dataset):
    """
    Build a dictionary associating words from a set of premises and
    hypotheses to unique integer indices.

    Args:
        dataset: A dictionary containing the premises and hypotheses for which
            a worddict must be built. The dictionary is assumed to have the
            same form as the dicts built by the 'read_data' function of this
            module.
        num_words: Integer indicating the maximum number of words to
            keep in the worddict. If specified, only the 'num_words' most
            frequent words will be kept. If set to None, all words are
            kept. Defaults to None.

    Returns:
        A dictionary associating words to integer indices.
    """
    """
    vocab = Vocabulary(num_words)
    for ins in dataset:
        for word in ins['premise']:
            vocab.add(word)
        for word in ins['hypothesis']:
            vocab.add(word)
    vocab.build_vocab()
    """

    vocab = Vocabulary(unknown='_OOV_', padding='_PAD_')
    dataset.apply(lambda x: [vocab.add(word) for word in x['premise']])
    dataset.apply(lambda x: [vocab.add(word) for word in x['hypothesis']])
    vocab.build_vocab()

    return vocab


def transform2idx(dataset, vocab):
    label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    dataset.drop(lambda x: x['label'] not in label_dict)
    dataset.apply(lambda x: [vocab.to_index(word) for word in x['premise']], new_field_name='premises')
    dataset.apply(lambda x: [vocab.to_index(word) for word in x['hypothesis']], new_field_name='hypotheses')
    dataset.apply(lambda x: label_dict[x['label']], new_field_name='label')
    dataset.apply(lambda x: len(x['premise']), new_field_name='premises_lengths')
    dataset.apply(lambda x: len(x['hypothesis']), new_field_name='hypotheses_lengths')

    dataset.set_input("premises", "premises_lengths", "hypotheses", "hypotheses_lengths")
    dataset.set_target("label")
    return dataset


def build_embedding_matrix(worddict, embeddings_file):
    """
    Build an embedding matrix with pretrained weights for a given worddict.

    Args:
        worddict: A dictionary associating words to unique integer indices.
        embeddings_file: A file containing pretrained word embeddings.

    Returns:
        A numpy matrix of size (num_words+4, embedding_dim) containing
        pretrained word embeddings (the +4 is for the padding, BOS, EOS and
        out-of-vocabulary tokens).
    """
    # Load the word embeddings in a dictionnary.
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf8') as input_data:
        for line in input_data:
            # raw_line = line
            line = line.split()

            try:
                # Check that the second element on the line is the start
                # of the embedding and not another word. Necessary to
                # ignore multiple word lines.
                float(line[1])
                word = line[0]
                if word in worddict:
                    embeddings[word] = line[1:]

            # Ignore lines corresponding to multiple words separated
            # by spaces.
            except ValueError:
                # print(raw_line)
                continue

    num_words = len(worddict)
    embedding_dim = len(list(embeddings.values())[0])
    # print(embedding_dim)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # Actual building of the embedding matrix.
    for word, i in worddict.items():
        if word in embeddings:
            embedding_matrix[i] = np.array(embeddings[word], dtype=float)
        else:
            if word == "_PAD_":
                continue
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_matrix[i] = np.random.normal(size=embedding_dim)

    return embedding_matrix


def preprocess_nli_data(inputdir,
                        embeddings_file,
                        targetdir):
    """
    Preprocess the data from some NLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, '*_train.txt'):
            train_file = file
        elif fnmatch.fnmatch(file, '*_dev.txt'):
            dev_file = file
        elif fnmatch.fnmatch(file, '*_test.txt'):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    dataset = read_data(os.path.join(inputdir, train_file))

    # print(dataset[0])
    # print(dataset[-1])

    print("\t* Computing worddict and saving it...")
    vocab = build_worddict(dataset)
    worddict = vocab.word2idx
    util.save_pickle(worddict, targetdir, "worddict.pkl")

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform2idx(dataset, vocab)
    print(transformed_data[0])
    print(transformed_data[-1])
    print("\t* Saving result...")
    util.save_pickle(transformed_data, targetdir, "train_data.pkl")

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    dataset = read_data(os.path.join(inputdir, dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform2idx(dataset, vocab)
    print("\t* Saving result...")
    util.save_pickle(transformed_data, targetdir, "dev_data.pkl")


    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")
    dataset = read_data(os.path.join(inputdir, test_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform2idx(dataset, vocab)
    print("\t* Saving result...")
    util.save_pickle(transformed_data, targetdir, "test_data.pkl")

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(worddict, embeddings_file)
    util.save_pickle(embed_matrix, targetdir, "embeddings.pkl")



if __name__ == "__main__":
    args = ConfigSection()
    ConfigLoader().load_config("../data/config.json", {"preprocess": args})
    train_data, dev_data, test_data= preprocess_nli_data(os.path.normpath(args["data_dir"]),
                        os.path.normpath(args["embeddings_file"]),
                        os.path.normpath(args["target_dir"]))