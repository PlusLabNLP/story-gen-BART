"""
A script that measures intra-story repetition based on repeated trigrams in a one line story.
Outputs the average of all stories in one file.
"""

import argparse
from collections import Counter

from numpy import mean, std


def get_ngrams(text, n=3, sent_delimiter="</s>"):
    """takes file with text and an optional sentence delimiter, returns counter of ngrams"""
    ngrams = Counter()
    sentences = [sent.split() for sent in text.strip().split(sent_delimiter)]  # nested list words in sents
    for sent in sentences:
        for i in range(len(sent) - n + 1):
            ngrams[' '.join(sent[i:i+n])] += 1
    return ngrams

def main(story_file):
    with open(story_file, "r") as infile:
        stories = infile.readlines()

    # for each story, get repetition metric
    repetition_array = []
    for story in stories:
        trigrams = get_ngrams(story)
        unique_trigrams = len(trigrams)
        total_trigrams = sum(trigrams.values())
        try:
            trigram_repetition = ((total_trigrams - unique_trigrams) / total_trigrams) * 100
        except ZeroDivisionError:
            trigram_repetition = 0
        repetition_array.append(trigram_repetition)
    return mean(repetition_array)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('story_file', type=str, help='filepath with generated stories, one per line')
    args = p.parse_args()

    # get stories first since only counting within one story
    with open(args.story_file, "r") as infile:
        stories = infile.readlines()

    # for each story, get repetition metric
    repetition_array = []
    for story in stories:
        trigrams = get_ngrams(story)
        unique_trigrams = len(trigrams)
        total_trigrams = sum(trigrams.values())
        try:
            trigram_repetition = ((total_trigrams - unique_trigrams) / total_trigrams) * 100
        except ZeroDivisionError:
            trigram_repetition = 0
        repetition_array.append(trigram_repetition)

    print("Intra-story trigram repetition for {} stories %:\n"
          "Mean: {:.2f} Min: {:.2f} Max: {:.2f} StDev {:.2f}".format(
        len(stories), mean(repetition_array),
        min(repetition_array), max(repetition_array),
        std(repetition_array)))
    print("-" * 89)

