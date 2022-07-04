import os
import json
import re
import argparse
import cProfile
import time
import json
import re
import io
import spacy
from tqdm import tqdm
import glob, os

def srl_resume(fin, fout):
    with open(fin, "r") as fin:
        data = json.load(fin)
        all_storyline_list = []
        for story in tqdm(data):
            labeled_doc = story["doc"]
            doc = []
            # print(labeled_doc)
            for token in labeled_doc:
                token = clean_token(token)
                doc.append(token)
            # print("Clean doc", doc)
            all_descriptions = story["srl"]
            clusters = story["clusters"]
            storyline = extract_storyline(doc, clusters, all_descriptions)
            # print(storyline)
            tmp2 = []
            for sent_des in storyline:
                tmp = []
                for _des in sent_des:
                    for tag, word in _des.items():
                        _des = ''.join(tag + ' ' + word)
                        tmp.append(_des)
                sent_storyline = ' # '.join(tmp)
                # print(sent_storyline)
                tmp2.append(sent_storyline)
                tmp3 = ' </s> '.join(tmp2)
            # print('\n')
            # print("ALLL", tmp3)
            all_storyline_list.append(tmp3)
            # print("="*89)
        all_storyline = '\n'.join(all_storyline_list)
        with open(fout, 'w') as fout:
            fout.write(all_storyline)


def clean_token(token):
    token = re.sub(r'(<ent> [0-9]*) | (</ent> [0-9]*)','', token)
    return token

def doc_resume(fin,fout):
    with open(fin, "r") as fin:
        data = json.load(fin)
        all_story_list = []
        for story in tqdm(data):
            labeled_doc = story["doc"]
            if len(labeled_doc) > 1:
                labeled_doc.pop()
                one_story = ' '.join(labeled_doc)
            all_story_list.append(one_story)
        all_story = '\n'.join(all_story_list)
    with open(fout, 'w') as fout:
        fout.write(all_story)

def story_resume(fin,fout):
    with open(fin, "r") as fin:
        data = json.load(fin)
        all_story_list = []
        for story in tqdm(data):
            labeled_doc = story["doc"]
            doc = []
            # print(labeled_doc)
            for token in labeled_doc:
                token = clean_token(token)
                doc.append(token)
                one_story = ' '.join(doc)
            all_story_list.append(one_story)
        all_story = '\n'.join(all_story_list)
    with open(fout, 'w') as fout:
        fout.write(all_story)

class Sentence(object):
    """docstring for Sentence"""
    def __init__(self, string, begin, end):
        super(Sentence, self).__init__()
        self.string = string
        self.begin = begin
        self.end = end


class Story(object):
    """docstring for Story"""
    def __init__(self, char_list):
        super(Story, self).__init__()
        self.char_list = char_list

    def join_sentence(self):
        """
        After using Allennlp coref pareser,
        use the word_tokenzied list from a whole story to join a sent_tokenized list,
        sep flag is </s>
        """
        idx = 0
        length = len(self.char_list)
        pre_idx = 0
        curent_string = ''
        sentences = []
        while idx < len(self.char_list):
            if self.char_list[idx] == '</s>' and idx + 1 < length:
            #if self.char_list[idx] == '<' and idx + 2 < length and self.char_list[idx + 1] == '/s' and self.char_list[idx + 2] == '>':
                sentence = Sentence(curent_string[:len(curent_string)-1], pre_idx, idx)
                sentences.append(sentence)
                curent_string = ''
                # pre_idx = idx = idx + 3
                pre_idx = idx = idx + 1
            else:
                curent_string = curent_string + self.char_list[idx] + " "
                idx += 1
        sentence = Sentence(curent_string[:len(curent_string)-1], pre_idx, idx)
        sentences.append(sentence)
        return sentences

def extract_storyline(doc, clusters, all_descriptions):
    """
    After getting all srl anf coref clusters, we need check if one ARG is in clusters, if so we need to change it to "ent{}"
    :param doc:
    :param clusters:
    :param srl_model:
    :param batch_size:
    :param cuda_device:
    :return:
    """
    document = Story(doc)
    sentences = document.join_sentence()
    text = " ".join(document.char_list)
    all_descriptions = all_descriptions
    storyline = []
    # print(len(sentences))
    # print(len(all_descriptions))
    if len(sentences) != len(all_descriptions):
        assert ("SRL WRONG, the length of sentence is not equal to length of descriptions")
    for s in sentences:
        descriptions = all_descriptions[sentences.index(s)]
        sentence_description = []
        for description in descriptions:
            items = re.findall(r"\[(.+?)\]+?", description)  # only context
            _description = {}
            for item in items:
                tag = item.split(": ")[0]
                if tag == "V":
                    _description["<V>"] = item.split(': ')[1]
                elif tag in ["ARG0", "ARG1", "ARG2"]:
                    new_argument = replace_ent(item, s, doc, clusters)
                    for i in range(0, 3):
                        if tag == "ARG{}".format(i):
                            _description["<A{}>".format(i)] = new_argument
            _description = compress(_description)
            # print("*****")
            # print(_description)
            # tmp.append(_description)
            # print("*****")

            if len(_description) > 0:
                sentence_description.append(_description)
                # storyline.append(" #")
        storyline.append(sentence_description)
    # print(storyline_add_demilt)
    return storyline

def intersection(list1, list2):
    """
    helper function to find wheter srl argument index overlap with coref_resolution clusters list
    :param list1:
    :param list2:
    :return: the intersection part of two list
    """
    l = max(list1[0], list2[0])
    r = min(list1[1],list2[1])
    if l > r:
      return []
    return [l, r]

def replace_ent(argument, sentence, doc, clusters):
    """
    comparing the srl results and coreference resolution results,
    and change "ARG{}" to "ent{}" if in clusters
    """
    sub_sentence = argument.split(': ')[1]
    sub_sentence_words = sub_sentence.split(' ')
    new_argument = ''
    begin = end = -1
    for i in range(sentence.begin, sentence.end - len(sub_sentence_words)):
        is_match = True
        for j in range(len(sub_sentence_words)):
            if sub_sentence_words[j] != doc[i + j]:
                is_match = False
                break
        if is_match:
            begin = i
            end = i + len(sub_sentence_words)
            break
    for ent_idx in range(len(clusters)):
        for ent_range in clusters[ent_idx]:
            intersection_range = intersection(ent_range, [begin, end])
            if len(intersection_range) > 0:
                for replace_idx in range(0, min(len(sub_sentence_words), intersection_range[1] - intersection_range[0] + 1)):
                    sub_sentence_words[replace_idx] = "ent {}".format(ent_idx)
    for i in range(len(sub_sentence_words)):
        if i == 0 or sub_sentence_words[i - 1] != sub_sentence_words[i]:
            new_argument += sub_sentence_words[i]
        else:
            continue
        if i != len(sub_sentence_words) - 1:
            new_argument += ' '
    return new_argument


def compress(sentence_description):
  # conpress very long and messy SRL output to more abstract
    new_dic = sentence_description
    #rule 1:Delete some lines which only have V, since SRL aim is to learn info like “who does what”, or “ who does what to who”,
    # if it only has a verb prediction, it’s useless
    if "<A0>" not in sentence_description and "<A1>" not in sentence_description and "<A2>" not in sentence_description:
        new_dic = {}
        return new_dic
    #rule 2:  Delete some lines whose Verb is “be” or modal verb.
    if sentence_description.get("<V>") in ["is", "was", "were", "are", "be", "\'s", "\'re", "\'ll",
                                           "can", "could", "must", "may", "have to", "has to",
                                           "had to", "will", "would", "has", "have", "had", "do", "does", "did"]:
        new_dic = {}
        return new_dic
    #rule 3: Delete some lines whose AGR length exceed 5, then delete that line.
    for i in range(0,3):
        if "<A{}>".format(i) in sentence_description and len(sentence_description.get("<A{}>".format(i)).split(" ")) > 5:
            new_dic = {}
            return new_dic
    return new_dic




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='type of train, string, test')
    args = parser.parse_args()

    dir = "data/writingPrompts/srl_resume/{}/".format(args.type)
    file_list = glob.glob("data/writingPrompts/srl_resume/{}/*.json".format(args.type), recursive=True)
    sort_file = sorted(file_list, key=lambda i: (os.path.split(i)[1].split('.')[-2]))
    # files = os.listdir(dir)
    # files_txt = [i for i in files if i.endswith('.json')]
    # files.sort()
    # print(files)
    for file in sort_file:
        file_name = os.path.split(file)[1]
        print('Processing {}'.format(file_name))
        srl_resume(file, os.path.join(dir, 'plot.{}.txt'.format(format(os.path.split(file)[1].split('.')[-2]))))
        print('Saved plot.{}.txt'.format(format(os.path.split(plotfile)[1].split('.')[-2])))
        # doc_resume(os.path.join(dir, file), os.path.join(dir, 'story.{}.txt'.format(file_name.split('.')[-2])))
        # print('Saved story.{}.txt'.format(file_name.split('.')[-2]))
        # story_resume(os.path.join(dir, file), os.path.join(dir, 'raw_story.{}.txt'.format(file_name.split('.')[-2])))
        # print('Saved raw story.{}.txt'.format(file_name.split('.')[-2]))