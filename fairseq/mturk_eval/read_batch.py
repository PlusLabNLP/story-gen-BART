import argparse
import csv
import itertools
import os
import sys
from collections import defaultdict
import numpy as np

from make_batch import RELEVANCE_HEADERS
from scipy import stats

ALL_TYPES = ["baseline", "discriminators"]

class Score:

    def __init__(self, source, total=0, correct=0, none=0):
        """has a source string and scores it tracks"""
        self.source = source
        self.correct = correct
        self.total = total
        self.none = none

    def __str__(self):
        return "Type: {}, Accuracy: {:.2f}, Nones: {:.2f}, Total: {}".format(self.source, self.get_accuracy(),
                                                                     self.get_nones(), self.total)

    def get_accuracy(self):
        if not self.total:
            return 0
        return self.correct/self.total

    def get_nones(self):
        if not self.total:
            return 0
        return self.none/self.total

def validate_same_num_scores(list_of_lists):
    lengths = set(map(len,list_of_lists))
    if len(lengths) > 1:
        print("Differing numbers of score length ({})".format(lengths), file=sys.stderr)
        return False
    else:
        return True


def check_any_correct(list_of_lists):
    """checks for "hard" experiment examples by seeing if all of them are False or Zero (e.g. no correct options)"""
    return True if any(itertools.chain(*list_of_lists)) else False


def get_ordered_experiment_scores(title_exp_scores, skip_hard=False, majority=False):
    """takes a nested dict of the thing that was the same in an experiment (like a title or person)
    and constructs a dict of experiment to scores_list that is aligned by the thing that was the same.
    For stat significance tests with non-independent results

    skip_hard throws out things that all participants got wrong, and majority compresses all scores into one majority vote"""
    exp2scores = defaultdict(list)
    hard_experiments = 0
    for title in title_exp_scores:
        complete_sample_set = validate_same_num_scores(title_exp_scores[title].values())
        any_correct = check_any_correct(title_exp_scores[title].values())
        #if len(title_exp_scores[title].values()) <4:
        #    print("missing info")
        #    continue
        if not any_correct:
            hard_experiments += 1
            if skip_hard:
                continue
        if not complete_sample_set and not majority:
            print("Problem title {} ".format(title), file=sys.stderr)
            continue
        for exp, scores in title_exp_scores[title].items():
            if majority:
                scores = [np.around(np.mean(scores))]
            exp2scores[exp].extend(scores)

    print("{} number of experiments were hard".format(hard_experiments), file=sys.stderr)
    return exp2scores

def process_title_matching_results(files: list):

    all_scores = {t: Score(t) for t in ALL_TYPES}
    # need to pair by title for statistical significance
    title_exp_scores = defaultdict(lambda: defaultdict(list))
    #title_exp_story = defaultdict(lambda: defaultdict(list))
    for file in files:
        with open(file, newline='') as csvfile:
            csv_dict = csv.DictReader(csvfile)
            for row in csv_dict:
                # Check if row Rejected:
                if row['AssignmentStatus'] == 'Rejected':
                    continue
                experiment = row["Input.experiment"]
                #story=row["Input.story"] # Used for if I need to see the story for reference
                choice = row["Answer.selected_title"]
                if choice == "true_title":
                    correct = 1
                    all_scores[experiment].correct += 1
                elif choice == "none":
                    #continue  #TODO for real support removing None
                    all_scores[experiment].none += 1
                    correct = 0
                else:
                    correct = 0
                all_scores[experiment].total += 1
                # make the title_exp_scores dict for stat significance, where true is 0 and false is 1
                title = row["Input.title_true"]
                title_exp_scores[title][experiment].append(correct)
                #title_exp_story[title][experiment] = row["Input.story"]
        ## Print Results
    for score in sorted(all_scores.values(), key=lambda s:s.source):
        print("Type: {}\n"
              "Accuracy: {:.2f} Nones: {:.2f} Total: {}".format(score.source,
                                                                score.get_accuracy(),
                                                                score.get_nones(),
                                                                score.total))

    # _titles, _bstories, _dstories = [], [], []
    # for key in title_exp_story:
    #     exp2story = title_exp_story[key]
    #     if len(exp2story.values()) < 2:
    #         continue
    #     _titles.append(key)
    #     _bstories.append(exp2story["baseline"])
    #     _dstories.append(exp2story["discriminators"])
    #
    # with open("story_titles", "w") as st, open("baseline_stories", "w") as bs, open("disc_stories", "w") as ds:
    #     st.write("\n".join(_titles))
    #     bs.write("\n".join(_bstories))
    #     ds.write("\n".join(_dstories))

    return title_exp_scores, all_scores.values()


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-f', dest='files', nargs='+', help='files to read in')
    p.add_argument('-d', dest='input_dir')
    p.add_argument('-t', dest='type', choices=['relevance', 'coherence', 'overall'])
    return p.parse_args()



if __name__ == "__main__":
    args = setup_argparse()

    if args.input_dir:
        with os.scandir(args.input_dir) as source_dir:
            files = sorted([file.path for file in source_dir if
                            file.is_file() and not file.name.startswith('.')])
    else:
        files = args.files

    if args.type == "relevance":

        title_exp_scores, raw_scores = process_title_matching_results(files)
        # get_titles_with_high_scores(title_exp_scores)
        # exp2scores = get_ordered_experiment_scores(title_exp_scores, skip_hard=args.skip_hard,
        #                                            majority=args.majority)
        #
        # if args.skip_hard or args.majority:
        #     raw_scores = [Score(exp, correct=val.count(1), total=len(val))
        #                   for exp, val in exp2scores.items()]
        # for score in sorted(raw_scores, key=lambda s: s.get_accuracy(), reverse=True):
        #     print(score)
        #
        # winner = max(raw_scores, key=lambda s: s.get_accuracy())
        #
        # for exp in sorted(exp2scores):
        #     stat, p = stats.wilcoxon(exp2scores[winner.source], exp2scores[exp])
        #     print("Experiment: {} Stat: {} P: {}".format(exp, stat, p))

    elif args.type == "overall":
        pass
        #"Input.story_1_source", "Input.story_2_source" for mappings




