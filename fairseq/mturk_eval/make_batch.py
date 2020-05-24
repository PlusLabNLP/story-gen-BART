
import random
import argparse
import csv
import re
import sys

import os


#### Globalish things
RELEVANCE_HEADERS = ["story", "title_1", "title_2", "title_3", "true_title", "source"]
OVERALL_HEADERS = ["title", "story_1", "story_2", "story_1_source", "story_2_source"]
RATING_HEADERS = ["title", "story_1", "story_2", "story_3", "story_4",
                  "story_1_source", "story_2_source", "story_3_source", "story_4_source"]
COHERENCE_HEADERS = ["story_1", "story_2", "true_story", "source", "title"]


def clean_stories(stories: list) -> list:
    new_stories = []
    for story in stories:
        new_story = re.sub("\s", " ", story)
        new_stories.append(new_story)
    return new_stories


def process_mturk_results(csv_dict, output_dict):
    rejections = 0
    used_titles = set()
    for row in csv_dict:
        if row['Reject']:
            rejections += 1
            continue
        title = row['Input.title'].strip()
        # split up the human entries when they are duplicates
        set_num = 1 if title in used_titles else 0
        used_titles.add(title)

        stories = row['Answer.MultiLineTextInput'].split('|')[:2] # only first two text entries are stories.
        stories = clean_stories(stories)

        output_dict[set_num][title].extend(stories)
    print("Num Rejections: {}".format(rejections), file=sys.stderr)


def get_random_samples(samples: list, forbidden: set, num: int=2) -> list:
    chosen = random.choices(range(len(samples)), k=num)
    while set(chosen) & forbidden:
        num_wrong = len(set(chosen) & forbidden)
        chosen = set(chosen) - forbidden
        new_choices = random.choices(range(len(samples)), k=num_wrong)
        chosen.update(new_choices)

    return [samples[x] for x in chosen]



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-t', dest='type', choices=['relevance', 'coherence', 'overall', "rating"])
    p.add_argument('--stories', nargs='+', type=str)
    p.add_argument('--shuffled_stories', nargs='+', type=str, help="for coherence, pre-shuffled stories, coindexed with ")
    p.add_argument('--titles', type=str)
    p.add_argument('--outfile', type=str)
    #p.add_argument('-e', dest='experiment', choices=["baseline", "discriminators"], help="experiment type")
    p.add_argument('--num_samples', type=int, default=100)
    p.add_argument('--randomize', action='store_true')

    #p.add_argument('-d', dest='inputdir', type=str)
    args = p.parse_args()

    # header_base = {"title": 'title_{}', "story": 'Story{}_{}'}
    # title_story_dict = {0: defaultdict(list), 1: defaultdict(list)} # two human sets for duplication
    with open(args.titles, "r") as title_in:
        titles = title_in.readlines()

    num_samples = min(args.num_samples, len(titles))

    #outdir = os.path.split(args.stories[0])[0] # TODO fix this
    out_csv = csv.writer(open(args.outfile, 'w', newline='', ))
    print("Working on {} lines in {}".format(num_samples, args.outfile))

    if args.type == "relevance" or args.type == "coherence":
        infile = args.stories[0] # only one if relevance
        source = os.path.split(infile)[1]
        with open(infile, "r") as story_in: # assume one file TODO fix this
            stories = story_in.readlines()

        if args.type == "relevance":
            out_csv.writerow(RELEVANCE_HEADERS)
            idxs = list(range(3))
            for i in range(num_samples):
                sample_titles = [titles[i]]
                sample_titles.extend(get_random_samples(titles, {i}, 2)) # 0 will be gold rest will be random
                if args.randomize:
                    random.shuffle(idxs)
                true_title = "title_{}".format(idxs.index(0))
                new_row = [stories[i].strip(), sample_titles[idxs[0]].strip(),
                           sample_titles[idxs[1]].strip(), sample_titles[idxs[2]].strip(),
                           true_title, source]
                out_csv.writerow(new_row)
        else:
            out_csv.writerow(COHERENCE_HEADERS)
            idxs = [0,1]
            with open(args.shuffled_stories[0], "r") as fin:
                shuffled = fin.readlines()
            for i in range(num_samples):
                these_stories = [stories[i].strip(), shuffled[i].strip()]
                if args.randomize:
                    random.shuffle(idxs)
                true_story = "story_{}".format(idxs.index(0))
                new_row = [these_stories[idxs[0]], these_stories[idxs[1]], true_story, titles[i].strip()]
                out_csv.writerow(new_row)



    elif args.type == "overall":
        out_csv.writerow(OVERALL_HEADERS)
        f1, f2 = args.stories
        with open(f1, "r") as fin1, open(f2, "r") as fin2:
            all_sources = os.path.split(f1)[1], os.path.split(f2)[1]
            all_stories = fin1.readlines(), fin2.readlines()
            idxs = list(range(2))
            for i in range(num_samples):
                if args.randomize:
                    random.shuffle(idxs)
                new_row = [titles[i].strip(), all_stories[idxs[0]][i].strip(), all_stories[idxs[1]][i].strip(),
                           all_sources[idxs[0]], all_sources[idxs[1]]]
                out_csv.writerow(new_row)

    elif args.type == "rating":
        out_csv.writerow(RATING_HEADERS)
        f1, f2, f3, f4 = args.stories
        with open(f1, "r") as fin1, open(f2, "r") as fin2, open(f3, "r") as fin3, open(f4, "r") as fin4:
            all_sources = os.path.split(f1)[1], os.path.split(f2)[1], \
                                                         os.path.split(f3)[1], os.path.split(f4)[1]
            all_stories = fin1.readlines(), fin2.readlines(), fin3.readlines(), fin4.readlines()
            lengths = [len(titles), len(all_stories[0]), len(all_stories[1]),
                       len(all_stories[2]), len(all_stories[3])]
            if len(set(lengths)) > 1:
                print("Warning: not same number of titles and stories: {}".format(lengths))
            idxs = list(range(4))
            for i in range(num_samples):
                if args.randomize:
                    random.shuffle(idxs)  # shuffle story order on each row
                new_row = [titles[i].strip(), all_stories[idxs[0]][i].strip(), all_stories[idxs[1]][i].strip(),
                           all_stories[idxs[2]][i].strip(), all_stories[idxs[3]][i].strip(),
                           all_sources[idxs[0]], all_sources[idxs[1]], all_sources[idxs[2]], all_sources[idxs[3]]]
                out_csv.writerow(new_row)











    # for file in files:
    #     mturk_results_data = csv.DictReader(open(file, newline=''))
    #     process_mturk_results(mturk_results_data, title_story_dict)
    # process_generated_results(args.genfile, title_story_dict)
    # write_csv_file(title_story_dict, args.outfile, header_base)

