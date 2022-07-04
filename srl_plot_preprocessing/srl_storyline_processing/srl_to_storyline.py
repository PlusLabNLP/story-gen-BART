""" accepts data of format Title <EOT> Story"""
# TO DO enable it to take stories and titles or just stories
import sys
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
# from semantic_role_labeler import SemanticRoleLabelerPredictor
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.predictors.coref import CorefPredictor
from allennlp.models.archival import load_archive
from tqdm import tqdm
import argparse
import cProfile
import time
import json
import re
import io
import spacy
import os

# import myownallennlp.allennlp.predictors as PaperClassifierPredictor
# from myownallennlp.allennlp.dataset_readers import SemanticScholarDatasetReader
# from myownallennlp.allennlp.models import AcademicPaperClassifier


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

def load_stories(infile):
    articles = []
    with open(infile) as inf:
        for line in inf:
            texts = ' '.join(line.strip().split('\t'))
            articles.append(texts)
    print('loaded %d stories!'%len(articles))
    return articles


def coref_resolution(text, CorefPredictor):
    """
    using Allennlp pretranied model to do coreference resolution
    :param text: a story
    :param coref_model: pretrained model weight, you should define its path in hyperpramenter
    :param cuda_device: if it >=0, it will load archival model on GPU otherwise CPU
    :return:  first return is a list of word_tokennize list of one story,
            second returen is a three layers list, [[[1,1],[3,5]],[6,6],[8,11]], same entity's index will be clusted together
    """
    result = CorefPredictor.predict_tokenized(text)
    return result.get("document"), result.get("clusters")


def SRL(text, SRLpredictor, batch_size):
    """
    :param text: a string of  story
    :param srl_model: pretrained model weight, you should define its path in hyperpramenter
    :param batch_size:
    :param cuda_device: if it >=0, it will load archival model on GPU otherwise CPU
    :return: all predictions after srl
    """

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = SRLpredictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = SRLpredictor.predict_batch_json(batch_data)
        return results

    batch_data = []
    all_predictions = []
    for line in text.split("</s>"):
        if not line.isspace():
            line = {"sentence":line.strip()}
            line = json.dumps(line)
            json_data = SRLpredictor.load_line(line)
            batch_data.append(json_data)
            # print(batch_data)
            if len(batch_data) == batch_size:
                predictions = _run_predictor(batch_data)
                # print("==========================")
                # print(len(predictions))
                all_predictions.append(predictions)
                batch_data = []
    if batch_data:
        predictions = _run_predictor(batch_data)
        all_predictions.append(predictions)
    all_description = []

    for batch in all_predictions:
        for sentence in batch:
            verbs = sentence.get("verbs")
            description = []
            for verb in verbs:
                description.append(verb.get("description"))
            all_description.append(description)
    return all_description

# range_list = parse(dec, doc, doc_current_index)
def extract_storyline(doc, clusters, SRLpredictor,batch_size):
    print(doc)
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
    all_descriptions = SRL(text,SRLpredictor,batch_size)
    storyline = []
    # print(len(sentences))
    # print(len(all_descriptions))
    if len(sentences) != len(all_descriptions):
        assert ("SRL WRONG, the length of sentence is not equal to length of descriptions")
    for s in sentences:
        descriptions = all_descriptions[sentences.index(s)]
        for description in descriptions:
            sentence_description = {}
            items = re.findall(r"\[(.+?)\]+?", description)  # only context
            for item in items:
                tag = item.split(": ")[0]
                if tag == "V":
                    sentence_description["<V>"] = item.split(': ')[1]
                elif tag in ["ARG0", "ARG1", "ARG2"]:
                    new_argument = replace_ent(item, s, doc, clusters)
                    for i in range(0, 3):
                        if tag == "ARG{}".format(i):
                            sentence_description["<A{}>".format(i)] = new_argument
            sentence_description = compress(sentence_description)
            if len(sentence_description) > 0:
                storyline.append(sentence_description)
                storyline.append("#")
        storyline.append("</s>")
    # print(storyline_add_demilt)
    return storyline, all_descriptions

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

def label_story(doc, cluster):
    for i, item in enumerate(cluster):
        for ent in item:
            beg = ent[0]
            end = ent[1]
            #TODO change this logic
            doc[beg] = "<ent> {0} {1}".format(i, doc[beg])
            doc[end] = "{0} </ent> {1}".format(doc[end], i)
    labeled_story = " ".join(doc)
    return labeled_story

def spacy_word_token(text,nlp):
    doc = nlp(text)
    token_list = [t.text for t in doc]
    return token_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to input file')
    parser.add_argument('--output_file', type=str, help='path to output file')
    parser.add_argument('--coref_model', type=str, help='path to pretrained model weight for corefrence resolution')
    parser.add_argument('--srl_model', type=str, help='path to pretrained mode weight for semantic role labeler')
    parser.add_argument('--batch', type=int, default=1, help='The batch size to use for processing')
    parser.add_argument('--cuda', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--save_coref_srl', type=str, help='dir for saving coref clusters and doc and srl for reusme')
    parser.add_argument('--label_story', type=str, help='dir for saving the stories after add ent label')
    parser.add_argument('--title', type=str, help='dir for saving the valid titles')
    # parser.add_argument('--reusem',  action='store true', help='reusem the coref, srl prediction')
    args = parser.parse_args()

    # Check GPU memeory.
    check_for_gpu(args.cuda)
    #load spacy tokenizer
    special_chars = {'<EOL>', '<EOT>', '<eos>', '</s>', '#', '<P>', "``", "\"", '[UNK]'} # harcoded list of special characters not to touch
    spacy_model = 'en_core_web_sm'
    # TODO add spacy check here to autodownload rather than error
    nlp = spacy.load(spacy_model)
    # Need to special case all special chars for tokenization
    for key in special_chars:
        nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])

    articles = load_stories(args.input_file)
    all_storyline = []
    all_prediction = []
    all_labeled_stories = []
    all_title = []

    all_json = []
    # Corefpredictor = CorefPredictor.from_archive(load_archive(args.coref_model, args.cuda))
    # for text in tqdm(articles):
    #     # try:
    #     #     title = text.split(" <EOT> ")[0]
    #     # except:
    #     #     print("title is empty, out of range")
    #     try:
    #         story = spacy_word_token(text, nlp)
    #         #story = spacy_word_token(text.split(" <EOT> ")[1], nlp)
    #     except:
    #         print("story is empty, out of range")
    #     try:
    #         doc, clusters = coref_resolution(story,Corefpredictor)
    #         all_json.append({"doc": doc, "clusters": clusters})
    #
    #     except RuntimeError:
    #         print("Runtime Error")
    #
    # with open("test_short.json", "w") as fout:
    #     json.dump(all_json, fout, ensure_ascii=False)


    # Now to SRL
    SRLpredictor = SemanticRoleLabelerPredictor.from_archive(load_archive(args.srl_model, args.cuda))
    with open("test_short.json", "r") as fin:
        docs_clusters = json.load(fin)
        for text in tqdm(docs_clusters):
            doc, clusters = text["doc"], text["clusters"]
            text_info = {}
            storyline, srl = extract_storyline(doc,clusters, SRLpredictor, args.batch)
            labeled_story = label_story(doc, clusters)

            if len(storyline) > 0:
                all_storyline.append(storyline)
                text_info["doc"] = doc
                text_info["clusters"] = clusters
                text_info["srl"] = srl
                all_prediction.append(text_info)
                all_labeled_stories.append(labeled_story)

    if len(all_storyline) > 0:
        print("Save {} storylines!".format(len(all_storyline)))
        with io.open(args.output_file, "w", encoding='utf8') as fout:
            json.dump(all_storyline, fout, ensure_ascii=False)

        # print("Save {} valid titles!".format(len(all_title)))
        # with io.open(args.title, "w", encoding='utf8') as fout:
        #     json.dump(all_title, fout, ensure_ascii=False)

        print("Save {} valid and labeled stories!".format(len(all_labeled_stories)))
        with io.open(args.label_story, "w", encoding='utf8') as fout:
            json.dump(all_labeled_stories, fout, ensure_ascii=False)


        print("Save {} coref and srl predictions!".format(len(all_prediction)))
        with io.open(args.save_coref_srl, "w", encoding='utf8') as fout:
            json.dump(all_prediction, fout, ensure_ascii=False)
