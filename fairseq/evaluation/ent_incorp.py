"""script that prints out misc metrics on story generations (or gold stories). Metrics:
* Vocab, Vocab:Token Ratio
* # Unique verbs
* % Diverse verbs (may not be valid as we remove really common ones)
* # unique entities per story
* length of generations
"""

import argparse
import re
from similarity.levenshtein import Levenshtein

from numpy import mean, std

#TODO clean this up and put it in utils. Do that for all the imports
from ordered_word_verb_incorp import split_file, read_file

def ent_incorp(plots, stories):
    # calculate entity incorporation in storyline and story
    # return average score of all lines
    entity_numbers = re.compile("(?<=\sent\s)\d+")
    ent_rate = []
    for plot, story in zip(plots, stories):
        plot_entities = list(entity_numbers.findall(plot))
        story_entities = list(entity_numbers.findall(story))
        levenshtein = Levenshtein()
        incorp_ent = levenshtein.distance(plot_entities, story_entities) # compute edit distance between the ent storyline and story
        try:
            ent_rate_each = incorp_ent/max(len(plot_entities), len(story_entities)) # normalise the distance
        except ZeroDivisionError:
            ent_rate_each = 0
        ent_rate.append(ent_rate_each)

        #print("Entity incorporation rate %:")
        #print("Mean: {:.2f} Min: {:.2f} Max: {:.2f} StDev {:.2f}".format(min(ent_rate), max(ent_rate),
        #                                                                 mean(ent_rate), std(ent_rate)))
    return mean(ent_rate), min(ent_rate), max(ent_rate), std(ent_rate)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--story_file', type=str,
                        help='location of story file. Can have either just stories or stories and plots')
    p.add_argument('--plot_file', type=str,
                        help='if given, assume plots are NOT in the story file')
    args = p.parse_args()

    if not args.plot_file:
        storylines, stories = split_file(args.story_file)
    else:
        storylines, stories = read_file(args.plot_file), read_file(args.story_file)

    results = ent_incorp(storylines, stories)
    print("Mean: {:.2f} Min: {:.2f} Max: {:.2f} StDev {:.2f}".format(*results))

