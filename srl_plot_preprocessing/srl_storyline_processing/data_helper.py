import argparse
import csv
import pandas as pd
import numpy as np

def load_stories(infile):
    articles = []
    with open(infile) as inf:
        for line in inf:
            texts = ' '.join(line.strip().split('\t'))
            articles.append(texts)
    print('loaded %d stories!'%len(articles))
    return articles

def check_number_of_sentence(story):
    length = len(story.split("</s>"))
    max = 0
    for sentence in story.split("</s>"):
        word_length = len(sentence.split(" "))
        if word_length > max:
            max = word_length
    return length,max

def count(stories):
    df = pd.DataFrame(
        {
            "number_of_sentence": [],
            "count": [],
            "max_form_which_story": [],
            "maxmium_sentence_length": []
        }
    )
    df[["number_of_sentence", "count", "max_form_which_story", "maxmium_sentence_length"]] = df[
        ["number_of_sentence", "count", "max_form_which_story", "maxmium_sentence_length"]].astype(int)
    for i, text in enumerate(stories):
        length, max = check_number_of_sentence(text)
        if ((df['number_of_sentence'] == length).any()) == False:
            # print("***************")
            # if length not in df['number_of_sentence']:
            df = df.append({"number_of_sentence": length,
                            "count": 1,
                            "max_form_which_story": i,
                            "maxmium_sentence_length": max},
                           ignore_index=True)
        else:
            df.loc[df['number_of_sentence'] == length, 'count'] = df.loc[df.number_of_sentence == length]['count'] + 1
            if max > df.loc[df['number_of_sentence'] == length]['maxmium_sentence_length'].item():
                df.loc[df['number_of_sentence'] == length, 'maxmium_sentence_length'] = max
                df.loc[df['number_of_sentence'] == length, 'max_form_which_story'] = i
            else:
                pass
    return df

def count_blank_line(stories):
    count = 0
    for story in stories:
        if len(story) == 0:
            count += 1
    print("Empty lines:", count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to input file')
    parser.add_argument('--output_file', type=str, help='path to output file')
    parser.add_argument('--check_all', action='store_true', help='check all sentence length')
    parser.add_argument('--check_empty', action='store_true', help='check emepty sentence count')
    args = parser.parse_args()
    stories = load_stories(args.input_file)
    if args.check_all:
        df = count(stories)
        df.to_csv(args.output_file, index=None, header=True)
    if args.check_empty:
        count_blank_line(stories)
