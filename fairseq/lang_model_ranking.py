import argparse
import os
import sys
import numpy as np

import torch
from fairseq.models.bart import BARTModel


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--data_type', type=str, choices=["plot", "story"])
    p.add_argument('--ranking', action="store_true", help="output ranking accuracy of x vs y pairs")
    p.add_argument('--both', action="store_true", help="output both ranking accuracy and mean scores")
    p.add_argument('--gen_data', nargs="+", help="generated files")
    p.add_argument('--cond_data', type=str, help="filename of conditional data")
    p.add_argument('--gold', type=str, help="gold stories, necessary for ranking accuracy")
    return p.parse_args()



if __name__ == "__main__":
    args = setup_argparse()
    # prints results so need to redirect
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    use_cuda = torch.cuda.is_available()
    title_plot_sep = "<EOT>"

    if args.data_type == "plot":
        ### load BART model
        bart = BARTModel.from_pretrained(
            'checkpoint_full/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='full'
        )
    elif args.data_type == "story":
        bart = BARTModel.from_pretrained(
            'checkpoint-fullstory/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='fullstory'
        )

    bart.eval()
    if use_cuda:
        bart.cuda()  # remove this line if not running with cuda
        bart.half()  # doesn't work with CPU

    if args.ranking or args.both:
        with open(args.gold, "r") as fin:
            gold_stories = fin.readlines()
            # if args.data_type == "story":  # story probabilities are the intermediate representation as well as the story rep
            #     gold_titles, gold_plots = [], []
            #     for line in gold_stories:
            #         gold_title, gold_plot = line.strip().split(title_plot_sep)
            #         gold_titles.append(gold_title)
            #         gold_plots.append(gold_plot)

    with open(args.cond_data, "r") as fin:
        cond_data = fin.readlines()
        # if args.data_type == "story": # story probabilities are the intermediate representation as well as the story rep
        #     titles, plots = [], []
        #     for line in cond_data:
        #         title, plot = line.strip().split(title_plot_sep)
        #         titles.append(title)
        #         plots.append(plot)


    for file in args.gen_data:
        print("Working on {}...".format(file))
        # run sequence scorer
        gold_win, gen_win = 0, 0
        all_scores, all_gold_scores = [], []
        with open(file, "r") as fin:
            for i, line in enumerate(fin):
                with torch.no_grad():
                    score = bart.score_sequence([cond_data[i].strip()], [line.strip()])
                    #print(score)
                    all_scores.append(score)
                    if args.ranking:
                        gold_score = bart.score_sequence([cond_data[i].strip()], [gold_stories[i].strip()])
                        all_gold_scores.append(gold_score)
                        if gold_score > score:
                            gold_win += 1
                        else:
                            gen_win += 1
        # if args.data_type == "story":
        #     plot_scores = []
        #     for i in range(len(titles)):
        #         with torch.no_grad():
        #             score = bart.score_sequence([titles[i].strip()], [plots[i].strip()])
        #             plot_scores.append(score)
        #     if args.ranking or args.both:
        #         gold_plot_scores = []
        #         for i in range(len(gold_titles)):
        #             with torch.no_grad():
        #                 gold_score = bart.score_sequence([gold_titles[i].strip()], [gold_plots[i].strip()])
        #                 gold_plot_scores.append(gold_score)


        if args.ranking or args.both:
            print("Generated ranked above gold: {:.2f}".format(gen_win/(gold_win+gen_win)))

            print("Mean conditional probability of gold sequences: {:.2f}".format(
                    np.mean(all_gold_scores)))

        else:
            print("Mean conditional probability of generated sequences: {:.2f}".format(np.mean(all_scores)))



