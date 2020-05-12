"""A script to measure the keyword incorporation metrics of stories and storylines"""

import argparse
from itertools import combinations

import numpy as np
import re
import sys
from tqdm import tqdm

#from pytorch_src.utils import read_w2v TODO work out why this sometimes breaks
def split_file(fin):
    print("Spliting file...")
    # Split story_out.txt to storyline file and story_file
    with open(fin, "r") as fin:
        lines = fin.readlines()
        storylines, stories, errors = [], [], []
        for line in lines:
            if not line:
                continue
            if line.count('<EOT>') == 1 and line.count('<EOL>') == 1: # filter some line which has 0 or one more <EOT>/<EOL>
                try:
                    plot_story = line.split(" <EOT> ")[1]
                    plot = plot_story.split(" <EOL> ")[0]
                    story = plot_story.split(" <EOL> ")[1]
                except IndexError:
                    errors.append(line)
                    continue
                storylines.append(plot)
                stories.append(story)
            else:
                errors.append(line)
    print('{} line from text ({:.2f}%) have not correct <EOT>/<EOL>'.format(
        len(errors), len(errors) * 100 / len(lines)), file=sys.stderr)
    return storylines, stories

def read_w2v(w2v_path, word2index, n_dims=300, unk_token="unk"):
    """takes tokens from files and returns word vectors
    :param w2v_path: path to pretrained embedding file
    :param word2index: Counter of tokens from processed files
    :param n_dims: embedding dimensions
    :param unk_token: this is the unk token for glove 840B 300d. Ideally we make this less hardcode-y
    :return numpy array of word vectors
    """
    print('Getting Word Vectors...', file=sys.stderr)
    vocab = set()
    # hacky thing to deal with making sure to incorporate unk tokens in the form they are in for a given embedding type
    if unk_token not in word2index:
        word2index[unk_token] = 0 # hardcoded, this would be better if it was a method of a class

    word_vectors = np.zeros((len(word2index), n_dims))  # length of vocab x embedding dimensions
    with open(w2v_path) as file:
        lc = 0
        for line in file:
            lc += 1
            line = line.strip()
            if line:
                row = line.split()
                token = row[0]
                if token in word2index or token == unk_token:
                    vocab.add(token)
                    try:
                        vec_data = [float(x) for x in row[1:]]
                        word_vectors[word2index[token]] = np.asarray(vec_data)
                        if lc == 1:
                            if len(vec_data) != n_dims:
                                raise RuntimeError("wrong number of dimensions")
                    except:
                        print('Error on line {}'.format(lc), file=sys.stderr)
                    # puts data for a given embedding at an index based on the word2index dict
                    # end up with a matrix of the entire vocab
    tokens_without_embeddings = set(word2index) - vocab
    print('Word Vectors ready!', file=sys.stderr)
    print('{} tokens from text ({:.2f}%) have no embeddings'.format(
        len(tokens_without_embeddings), len(tokens_without_embeddings)*100/len(word2index)), file=sys.stderr)
    print('Tokens without embeddings: {}'.format(tokens_without_embeddings), file=sys.stderr)
    print('Setting those tokens to unk embedding', file=sys.stderr)
    for token in tokens_without_embeddings:
        word_vectors[word2index[token]] = word_vectors[word2index[unk_token]]
    return word_vectors


def get_tokens(lists):
    """take a list of filepaths, returns word2idx dict"""
    print('Getting tokens ... ...', file=sys.stderr)
    all_tokens = set()
    for list in lists:
        all_tokens.update(set(''.join(list).strip().split()))
    word2index = dict(map(reversed, enumerate(all_tokens)))
    return word2index


def cos_sim(v1, v2):
    return v1.dot(v2) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2)))


def cos_sim_array(vec, vec_array):
    """
    take dot product of 2 vectors. which reduces dimensionality and gives me an array of results.
    IMPORTANT that vec_array is first arg as a result
    :param vec: a vector
    :param vec_array: an array of vectors
    :return: cosine_sim_array of the cosine similarity between the vector and each vector in the array
    """
    dot_prod_array = np.dot(vec_array, vec)
    # print(dot_prod_array)
    len_vec_array, len_x_d = (vec_array**2).sum(axis=1) ** .5, (vec ** 2).sum() ** .5
    cosine_sim_array = np.divide(dot_prod_array, len_vec_array*len_x_d)
    return cosine_sim_array


def remove_chars(text: str, remove='#') -> str:
    """take a string and optional chars to remove and returns string without them"""
    return re.sub(r'[{}]'.format(remove), '', text)


def make_vec_array(word_list: list, word_vectors, word2index: dict, drop_set={'<EOL>', '<EOT>', '<V>', '<A2>', '<P>', '#', '[PAD]', '<A0>', '</s>', '<bos>', '<A1>', '<eos>', 'ent', '<ent>', '</ent>'}):
    """take a list of strings, an array of word vectors, return a numpy array of word vectors"""
    vecs = [np.array(word_vectors[word2index.get(word, 0)])
            for word in word_list if word not in drop_set]
    return np.array(vecs)

def word_verb_calc_similarity(storylines, stories, word2index, word_vectors):
    print('Calculating word and verb ordered incorporation...')
    """calculates cosine similarity between keywords in storyline and between keywords in storyline
    and corresponding sentence in story. Averaged over all """
    # clean storyline and story
    for i, line in enumerate(storylines):
        line = re.sub(r'ent [0-9]*', '', line)  # remove `ent XX ` in storyline
        processed_line = remove_chars(line).strip().split()
        storylines[i] = processed_line
    for i, line in enumerate(stories):
        line = re.sub(r'(<ent> [0-9]*) | (</ent> [0-9]*)', '', line)  # remove `<ent> XX` or `</ent> XX` in story
        processed_line = remove_chars(line).strip().split()
        stories[i] = processed_line

    #check alignment
    num_storylines = len(storylines)
    assert(num_storylines == len(stories)), "Mismatch between number of storylines and number of stories"

    # loop through stories and storylines and calc similarities
    word_incorporation_rate = 0
    verb_incorporation_rate = 0
    for i in range(num_storylines): # one_line refer to one storyling or story
        splited_storyline = ' '.join(storylines[i]).split('</s>') # sentenize storyline
        splited_story = ' '.join(stories[i]).split('</s>') # sentenize story
        all_storyline_word_array, all_storyline_verb_array, all_story_word_array = [], [], []
        # Processing storyline sentence by sentence
        for storyline in splited_storyline: # get verc for each sentence
            # make word vector array for storyline sentence by sentence
            storyline_word_array = make_vec_array(storyline.split(), word_vectors, word2index)  # one seentence storyline vectors
            all_storyline_word_array.append(storyline_word_array)  # nested list: all storyline vectors
            # finds all verbs in each sentence
            verbs = []
            for j, word in enumerate(storyline.split()):
                if word == '<V>':
                    try:
                        verbs.append(storyline.split()[j + 1])
                    except IndexError:
                        continue
            # make verb vector array for storyline sentence by sentence
            storyline_verb_array = make_vec_array(verbs, word_vectors, word2index)  # one seentence storyline vectors
            all_storyline_verb_array.append(storyline_verb_array)  # nested list: all storyline vectors

        # Processing story sentence by sentence
        for story in splited_story:
            # make word vector array for story sentence by sentence
            story_word_array = make_vec_array(story.split(), word_vectors, word2index)  # one sentence word vectors in story
            all_story_word_array.append(story_word_array)

        # calculate the similarities between the word/verb and the sentence
        valid_num_words_in_storyline = 0
        valid_num_verbs_in_storyline = 0
        this_word_incorporation_rate = 0
        this_verb_incorporation_rate = 0
        min_lenght = min(len(splited_story), len(splited_storyline)) # truncate storyline or story since they have different length
        for k in range(min_lenght):
            # compute word incoporation
            valid_num_words_in_storyline += len(all_storyline_word_array[k])
            for kw_vec in all_storyline_word_array[k]:
                try:
                    cosine_max = np.nanmax(cos_sim_array(kw_vec, all_story_word_array[k])) # compute cosine max with word and specific sentence
                    this_word_incorporation_rate += cosine_max
                except:
                    continue
            # compute verb incoporation
            valid_num_verbs_in_storyline += len(all_storyline_verb_array[k])
            for kw_vec in all_storyline_verb_array[k]:
                try:
                    cosine_max = np.nanmax(cos_sim_array(kw_vec, all_story_word_array[k]))  # compute cosine max with word and specific sentence
                    this_verb_incorporation_rate += cosine_max
                except:
                    continue
        try:
            word_incorporation_rate += this_word_incorporation_rate/valid_num_words_in_storyline
        except:
            word_incorporation_rate += 0 # when valid_num_words_in_storyline == 0
        try:
            verb_incorporation_rate += this_verb_incorporation_rate / valid_num_verbs_in_storyline
        except ZeroDivisionError:  # aviod some storylines have no verb, so the num_words_in_storyline will be 0
            verb_incorporation_rate += 0

    # report average over all in set
    word_incorporation_rate /= num_storylines
    verb_incorporation_rate /= num_storylines
    print('Metrics for {} samples'.format(num_storylines))
    print('ordered word_incorporation_rate : {:.2f} %'.format(word_incorporation_rate * 100))
    print('ordered verb_incorporation_rate : {:.2f} %'.format(verb_incorporation_rate * 100))
    return verb_incorporation_rate * 100, word_incorporation_rate * 100


def main(wordvec_file, storyline_file,story_file):
    word2idx = get_tokens([storyline_file, story_file])
    word_vectors = read_w2v(wordvec_file, word2idx)
    word_incorporation_rate, verb_incorporation_rate = word_verb_calc_similarity(storyline_file, story_file, word2idx, word_vectors)
    return word_incorporation_rate, verb_incorporation_rate


def read_file(filepath):
    with open(filepath, "r") as fin:
        return fin.readlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--story_file', type=str, help='location of story file. Can have either just stories or stories and plots')
    parser.add_argument('--plot_file', type=str, help='if given, assume plots are NOT in the story file')
    parser.add_argument('--wordvec_file', type=str, help='path to glove wordvec file' )
    args = parser.parse_args()

    if not args.plot_file:
        storylines, stories = split_file(args.story_file)
    else:
        storylines, stories = read_file(args.plot_file), read_file(args.story_file)
    word2idx = get_tokens([storylines, stories]) # takes list of arbitrarily many files
    word_vectors = read_w2v(args.wordvec_file, word2idx)
    word_verb_calc_similarity(storylines, stories, word2idx, word_vectors)


