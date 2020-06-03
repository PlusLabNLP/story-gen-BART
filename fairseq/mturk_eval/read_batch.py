import argparse
import csv
import itertools
import os
import re
import sys
from collections import defaultdict
from typing import Tuple

import numpy as np

from make_batch import RELEVANCE_HEADERS
from scipy import stats

ALL_TYPES = ["baseline", "discriminators"]
BANNED_WORKERS = {"A3S26SUAEYLPGH"}#, "A153HLAVH5FILS"}

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

class Rating:

    def __init__(self, source):
        """has a source string and scores it tracks"""
        self.source = source
        self.ratings = defaultdict(list)
        self.better_story = 0
        self.total = 0

    def percent_better_story(self):
        if self.total == 0:
            return 0
        return (self.better_story/self.total) *100

    def get_avg_rating(self):
        return {key: np.mean(values) for key, values in self.ratings.items()}

    def get_avg_rating_str(self):
        return "\n".join(["Mean {}: {:.2f}".format(key, np.mean(values)) for key, values in self.ratings.items()])

    def get_avg_var_rating_str(self):
        return "\n".join(["Mean {}: {:.2f} Var {:.2f}".format(key, np.mean(values), np.var(values))
                          for key, values in self.ratings.items()])

    def get_variance_str(self):
        return "\n".join(["Variance {}: {:.2f}".format(key, np.var(values)) for key, values in self.ratings.items()])

    def get_overall(self):
        return np.mean(self.ratings["overall_quality"])

    def get_relevance(self):
        return np.mean(self.ratings["relevance"])

    def get_interestingness(self):
        return np.mean(self.ratings["interestingness"])

    def get_coherence(self):
        return np.mean(self.ratings["coherence"])


def validate_same_num_scores(list_of_lists, verbose=False):
    lengths = set(map(len,list_of_lists))
    if len(lengths) > 1 and verbose:
        print("Differing numbers of score length ({})".format(lengths), file=sys.stderr)
        return False
    else:
        return True


def check_any_correct(list_of_lists):
    """checks for "hard" experiment examples by seeing if all of them are False or Zero (e.g. no correct options)"""
    return True if any(itertools.chain(*list_of_lists)) else False


def get_ordered_experiment_scores(title_exp_scores, skip_hard=False, majority=False, take_mean=False,
                                  missing=set()):
    """takes a nested dict of the thing that was the same in an experiment (like a title or person)
    and constructs a dict of experiment to scores_list that is aligned by the thing that was the same.
    For stat significance tests with non-independent results

    skip_hard throws out things that all participants got wrong, and majority compresses all scores into one majority vote"""
    exp2scores = defaultdict(list)
    hard_experiments = 0
    for title in title_exp_scores:
        if title in missing:
            continue
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
                if not take_mean:
                    if scores.count(0) == scores.count(1) and len(set(scores)) > 1:
                        print("ties on title {}".format(title))
                    scores = [np.around(np.mean(scores))]
                else:
                    scores = [np.mean(scores)]
            exp2scores[exp].extend(scores)

    print("{} number of experiments were hard (no participants got correct)".format(hard_experiments), file=sys.stderr)
    return exp2scores

# def process_story_ranking_data(files):
#     type_metric_scores = defaultdict(lambda: defaultdict(list))
#     title_type_metric_scores = defaultdict(lambda: defaultdict(list))
#     winners = []
#     for file in files:


def get_and_print_stats(exp2scores, exclude_zeros=False):
    """

    :param exp2scores: dict of experiment name (as a tuple) to array of scores
    :return: a list of tuples of experiment name (as a typle) and mean of scores
    """
    print("-" * 89)
    print("All Experiment Means:\n")
    sorted_exp = sorted(exp2scores.items())
    all_results = []
    normal, not_normal, alpha = 0, 0, 0.05
    for exp, scores in sorted_exp:
        # This is only needed if there are invalid scores
        #filtered_scores = list(filter(is_valid_score, scores)) # gets rid of empty string when people didn't reply
        if exclude_zeros:
            filtered_scores = list(filter(bool, scores))
        #print(sorted(filtered_scores))
            exp_mean = np.mean(filtered_scores)
        else:
            exp_mean = np.mean(scores)
        all_results.append((exp, exp_mean))

        # check shape of distribution
        k2, p = stats.normaltest(scores)
        if p < alpha:
            not_normal += 1
        else:
            normal += 1

    all_results.sort()
    for exp, exp_mean in all_results:
        print("Mean of {} : {:.2f}".format(exp, exp_mean))
    print("{} of distributions were normal and {} were not normal".format(normal, not_normal))

    return all_results

def get_best_scores_per_metric(result_means):
    """

    :param result_means: a list of (exp_type, metric) : mean, where mean is a float
    :return: a list of the best scores per metric/category
    """
    metric_list = set([result_pair[0][1] for result_pair in result_means]) # since it is a tuple with a tuple at idx 0 of (exp, metric)
    print(metric_list)
    best_experiments = []
    for metric in metric_list:
        means_per_metric = filter(lambda x: x[0][1] == metric, result_means)
        best = max(means_per_metric, key=lambda x: x[1])
        #print(best)
        best_experiments.append(best[0])

    return best_experiments

def stat_sig(winner, exp2scores, test_type, alpha=0.05, paired=False):
    """

    :param winner: tuple of win name and array of win scores
    :param exp2scores: exp2scores format (dictionary with key to array of values)
    :param test_type: a scipy.stats test
    :param alpha: threshold for significance
    :param paired: whether the test is a paired test and have to make sure everything matches
    :return: None
    """
    print('_'*89)
    print("Win type: {}".format(winner))
    win_scores = exp2scores[winner]
    not_equal_exp, equal_exp = [], []
    for exp, scores in exp2scores.items():
        if exp[1] != winner[1]: # make sure metric type matches...this should so be object oriented
            continue
        if paired: #filter both lists down so that they have valid values in both
            assert len(win_scores) == len(scores), "the lists of scores need to be of equal length for a paired test"
            score_diffs = []
            for i in range(len(win_scores)):
                if not win_scores[i] or not scores[i]: # this assumes the unset value of the pair is bool False. Won't work if zero is a valid value
                    continue
                else:
                    score_diffs.append(win_scores[i]-scores[i])
            print(score_diffs)
            stat, p = test_type(score_diffs)
        else:
            stat, p = test_type(win_scores, scores)
        if p < alpha:
            not_equal_exp.append((exp, p))
            #print("Distribution of winner vs. {} is not equal, p = {:.2f}".format(exp, p))
        else:
            equal_exp.append((exp,p))
            #print("Distribution of winner vs. {} is equal, p = {:.2f}".format(exp, p))
    print("{} are equivalent and {} are not".format(len(equal_exp), len(not_equal_exp)))
    print("Equal:")
    for x in equal_exp:
        print(x)
    print("Not Equal:")
    for x in not_equal_exp:
        print(x)

def get_choices(user_choice: str, correct_choice:str) -> Tuple[int, int]:
    # Note: correct choice *should* be 1 indexed but is zero indexed so this fixed
    user_int = re.findall(r"\d", user_choice)
    if user_int:
        user_int = int(user_int[0])
    correct_int = int(re.findall(r"\d", correct_choice)[0]) + 1
    return user_int, correct_int



def process_accuracy_results(files: list, experiment: str, confidence_threshold: int=0,
                             filter_attention=False, mapping: dict=None, pairwise_type=None):
    all_scores = {}
    # need to pair by title for statistical significance
    title_exp_scores = defaultdict(lambda: defaultdict(list))
    if experiment == "rating":
        title_exp_scores_i, title_exp_scores_o = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        title_exp_scores_r, title_exp_scores_c = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    #title_exp_story = defaultdict(lambda: defaultdict(list))
    for file in files:
        with open(file, newline='') as csvfile:
            csv_dict = csv.DictReader(csvfile)
            for row in csv_dict:
                # Check if row Rejected:
                if row['AssignmentStatus'] == 'Rejected' or row["WorkerId"] in BANNED_WORKERS or row["Reject"]:
                    continue
                if filter_attention and row.get("Answer.attn_check", "") != row.get("Input.true_answer", ""):
                    continue
                if confidence_threshold and experiment not in ["relevance", "rating"] :
                    if int(row["Answer.user_confidence"]) < confidence_threshold:
                        continue
                if experiment == "rating":
                    all_stories = ["story_{}".format(num) for num in range(1,5)] # currently not using 5th story
                    title = row["Input.title"]
                    better_story = row["Answer.better_story"]
                    revised_ratings = True if row["Answer.revised_ratings"] == "yes" else False
                    for story_num in all_stories:
                        exp_source = row["Input.{}_source".format(story_num)]
                        if mapping:
                            exp_source = mapping.get(exp_source, exp_source)
                        if exp_source not in all_scores:
                            all_scores[exp_source] = Rating(exp_source) # ratings don't have accuracies
                        i_score, o_score = int(row["Answer.{}_interestingness".format(story_num)]), \
                                           int(row["Answer.{}_quality".format(story_num)])
                        r_score, c_score = int(row["Answer.{}_relevance".format(story_num)]), \
                                           int(row["Answer.{}_coherence".format(story_num)])
                        all_scores[exp_source].ratings["interestingness"].append(i_score)
                        all_scores[exp_source].ratings["overall_quality"].append(o_score)
                        all_scores[exp_source].ratings["relevance"].append(r_score)
                        all_scores[exp_source].ratings["coherence"].append(c_score)
                        all_scores[exp_source].total += 1
                        if better_story == story_num:
                            all_scores[exp_source].better_story += 1

                        title_exp_scores_i[title][exp_source].append(i_score)
                        title_exp_scores_o[title][exp_source].append(o_score)
                        title_exp_scores_r[title][exp_source].append(r_score)
                        title_exp_scores_c[title][exp_source].append(c_score)
                        #title_exp_scores[title][exp_source]["revised_ratings"].append(revised_ratings) # todo this is a property of the title not the source but meh just recording it for now

                elif experiment == "relevance" or experiment == "coherence":
                    exp_source = row["Input.source"]
                    if mapping:
                        exp_source = mapping.get(exp_source, exp_source)
                    # add exp source to score dict
                    if exp_source not in all_scores:
                        all_scores[exp_source] = Score(exp_source)
                    # story=row["Input.story"] # Used for if I need to see the story for reference
                    if experiment == "relevance":
                        user_choice, correct_choice = get_choices(row["Answer.selected_title"], row["Input.true_title"])
                        title = row["Input.title_{}".format(correct_choice)]
                    else:
                        user_choice, correct_choice = get_choices(row["Answer.user_choice"], row["Input.true_story"])
                        title = row["Input.title"]

                    if user_choice == correct_choice:
                        correct = 1
                        all_scores[exp_source].correct += 1
                    elif not user_choice: # if it was None
                        all_scores[exp_source].none += 1
                        correct = 0
                    else:
                        correct = 0

                    all_scores[exp_source].total += 1
                    # make the title_exp_scores dict for stat significance
                    title_exp_scores[title][exp_source].append(correct)
                    # title_exp_story[title][experiment] = row["Input.story"]

                elif experiment == "overall":
                    title = row["Input.title"]
                    both_sources = {row["Input.story_1_source"], row["Input.story_2_source"]}
                    user_choice = row["Answer.user_choice"]
                    user_choice_source = row["Input.{}_source".format(user_choice)] if user_choice != "none" else None # this will be bogus if answer is none but it doesn't matter
                    if mapping:
                        user_choice_source = mapping.get(user_choice_source, user_choice_source)
                        both_sources = {mapping.get(exp_source, exp_source) for exp_source in both_sources}
                        if pairwise_type not in both_sources:
                            continue
                    for exp_source in both_sources:
                        if exp_source not in all_scores:
                            all_scores[exp_source] = Score(exp_source)
                        if user_choice == "none":
                            all_scores[exp_source].none += 1
                            correct = 0
                        elif exp_source == user_choice_source:
                            all_scores[exp_source].correct += 1
                            correct = 1
                        else:
                            correct = 0

                        all_scores[exp_source].total += 1
                        # make the title_exp_scores dict for stat significance
                        title_exp_scores[title][exp_source].append(correct)
                        # title_exp_story[title][experiment] = row["Input.story"]
    if experiment == "rating":
        for score in sorted(all_scores.values(), key=lambda s: s.source):
            print("-"*89)
            print("Type: {}\n"
                  "Average Rating: \n{} \nBetter Story: {:.2f}".format(score.source,
                                                                    score.get_avg_var_rating_str(),
                                                                    score.percent_better_story()))

        return (title_exp_scores_o, title_exp_scores_i, title_exp_scores_r, title_exp_scores_c), all_scores

    else:
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


def majority_vote(title_exp_scores):
    #all_sources = set([key for key in title_exp_scores.values()])
    all_scores = {} # Score(source) for source in all_sources} #TODO this might be dicts...
    incomplete, too_many = 0, 0
    for title in title_exp_scores:
        exp2score = title_exp_scores[title]
        for exp, scores in exp2score.items():
            if exp not in all_scores:
                all_scores[exp] = Score(exp)
            if len(scores) != 3:
                if len(scores) < 3:
                    incomplete += 1
                    continue
                if len(scores) > 3:
                    if scores.count(0) == scores.count(1) and len(set(scores)) > 1:
                        print(scores)
                        print("title in too many: {}".format(title))
                        too_many += 1
                        continue
            majority_score = stats.mode(scores).mode[0]
            all_scores[exp].correct += majority_score
            all_scores[exp].total += 1

    print("{} had less than 3 and {} more than 4".format(incomplete, too_many))
    print("\nFiltered by majority vote...")
    for score in sorted(all_scores.values(), key=lambda s:s.source):
        print("Type: {}\n"
              "Win Rate: {:.2f} Total: {}".format(score.source,
                                                                score.get_accuracy(),
                                                                score.total))

def process_rating_results(files):
    pass


def validate_same_titles(title_exp_scores):
    print("Total Titles: {}".format(len(title_exp_scores)))
    all_experiments = {k for key in title_exp_scores for k in title_exp_scores[key]}
    exp2missing = defaultdict(list)
    for t in title_exp_scores:
        these_exp = set(title_exp_scores[t].keys())
        for exp in all_experiments:
            if exp not in these_exp:
                exp2missing[exp].append(t)
    for exp in exp2missing:
        print("{} missing titles:".format(exp))
        print(exp2missing[exp])

    return set(itertools.chain(*exp2missing.values()))


def failed_attention_checks(files: list, include_rejected=False):
    #total, failed = 0,0
    failed2answer = {}
    for file in files:
        total, failed = 0, 0
        with open(file, newline='') as csvfile:
            csv_dict = csv.DictReader(csvfile)
            for row in csv_dict:
                title = row.get("Input.title", "ratings")
                if not include_rejected:
                    if row['AssignmentStatus'] == 'Rejected' or row["WorkerId"] in BANNED_WORKERS or row["Reject"]:
                        continue
                if row.get("Answer.attn_check", "") != row.get("Input.true_answer", ""):
                    failed += 1
                    failed2answer[title] = [row.get("Input.attn_question"), row.get("Input.true_answer")]
                total += 1
        print(file)
        print("Failed attention checks: {} of {} total ({:.2f})%".format(failed, total, failed/total*100))
    #for f, q in failed2answer.items():
    #    print("{} | {} | {}".format(f, *q))


def get_confidences(files, include_rejected=False, low_thresh=3):
    title2conf = defaultdict(list)
    for file in files:
        with open(file, newline='') as csvfile:
            csv_dict = csv.DictReader(csvfile)
            for row in csv_dict:
                title = row["Input.title"]
                if not include_rejected:
                    if row['AssignmentStatus'] == 'Rejected' or row["WorkerId"] in BANNED_WORKERS or row["Reject"]:
                        continue
                conf = int(row["Answer.user_confidence"])
                title2conf[title].append(conf)
    low_conf_titles = {t: np.mean(c) for t, c in title2conf.items() if np.mean(c) < low_thresh}
    print("Num titles with low conf (<{}): {} of {}".format(low_thresh, len(low_conf_titles), len(title2conf)))
    for t,c in low_conf_titles.items():
        print("{}: {:.2f}".format(t, c))


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-f', dest='files', nargs='+', help='files to read in')
    p.add_argument('-d', dest='input_dir')
    p.add_argument('-t', dest='type', choices=['relevance', 'coherence', 'overall', "rating", "confidence", "attn_check"])
    p.add_argument('--confidence_threshold', type=int, default=0, help="throw out scores with confidence lower than this number")
    p.add_argument('--attention', action="store_true", help="if there were attention checks, filter fails")
    p.add_argument('--pairwise_type', choices=["prompt2story", "no_disc"])
    return p.parse_args()



if __name__ == "__main__":
    args = setup_argparse()

    mapping = {
        "prompt_plot.human.story.human_readable.clean.filtered": "prompt2story",
        "prompt_plot.human.story.human_readable.clean.new_filtered": "prompt2story",
        "prompt_plot.auto.human_readable": "prompt2story",
        "test.txt.title.human.every_dedup_all_5.tsv.gen.story.human_readable.filtered": "disc",
        "test.txt.title.human.every_dedup_all_5.tsv.gen.story.human_readable.new_filtered": "disc",
        "test.txt.title.auto.every_dedup_all_5.tsv.gen.human_readable": "disc",
        "test.txt.title.human.gen.story.human_readable.clean.filtered": "no_disc",
        "test.txt.title.human.gen.story.human_readable.clean.new_filtered": "no_disc",
        "test.txt.title.auto.gen.human_readable": "no_disc",
        "tacl-human.txt.human_readable.clean.new_filtered": "tacl",
        "tacl-human.txt.human_readable.clean.filtered": "tacl",
        "tacl-auto.txt.human_readable": "tacl",
        "acl_baseline.human.clean.filtered": "acl",
        "acl_baseline.human.clean.new_filtered": "acl",
        "acl_baseline.auto.human_readable": "acl"

    }

    if args.input_dir:
        with os.scandir(args.input_dir) as source_dir:
            files = sorted([file.path for file in source_dir if
                            file.is_file() and not file.name.startswith('.')])
    else:
        files = args.files

    if args.type == "confidence":
        get_confidences(files)

    elif args.type == "attn_check":
        failed_attention_checks(files)


    else:

        title_exp_scores, raw_scores = process_accuracy_results(files, args.type,
                                                                args.confidence_threshold,
                                                                args.attention, mapping,
                                                                args.pairwise_type)

        failed_attention_checks(files)
        if args.type != "rating":
            missing_title_set = validate_same_titles(title_exp_scores)
            majority_vote(title_exp_scores)
            # get_titles_with_high_scores(title_exp_scores)
            exp2scores = get_ordered_experiment_scores(title_exp_scores, skip_hard=False,
                                                       majority=True, missing=missing_title_set)

        else:
            for m in title_exp_scores:
                print("Titles in this metric: {}".format(len(m)))
            # get_titles_with_high_scores(title_exp_scores)
            metrics2exp = []

            #TODO: Note that this below can't run stat sig becuase for some reason it doesn't do the "validate same num titles" thing accurately
            # for metric_dict in title_exp_scores: # it will be a tuple of len num metrics
            #     metrics2exp.append(get_ordered_experiment_scores(metric_dict, skip_hard=False,
            #                                            take_mean=True, majority=True))

        if args.type != "rating":
            winner = max(raw_scores, key=lambda s: s.get_accuracy())

            for exp in sorted(exp2scores):
                if exp == winner.source:
                    continue
                print(len(exp2scores[winner.source]))
                stat, p = stats.wilcoxon(exp2scores[winner.source], exp2scores[exp])
                print("Experiment: {} Stat: {} P: {}".format(exp, stat, p))
        else:
            target = "disc"
            for exp2scores in metrics2exp:
                for exp in sorted(exp2scores):
                    if exp == target:
                        continue
                    stat, p = stats.wilcoxon(exp2scores[target], exp2scores[exp])
                    print("Experiment: {} Stat: {} P: {}".format(exp, stat, p))


