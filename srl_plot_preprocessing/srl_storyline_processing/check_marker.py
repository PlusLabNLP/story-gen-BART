import csv
import numpy as np
import os
import argparse
import pandas as pd

def check(plotfile, storyfile, csvfile, count=0):
    with open(plotfile, 'r') as fin:
        plots = fin.readlines()
        print('plot : {} lines'.format(len(plots)))

    with open(storyfile, 'r') as fin:
        storys = fin.readlines()
        print('story : {} lines'.format(len(storys)))
    count == 0
    idx_list = []
    plot_list = []
    story_list = []
    for idx, _plot in enumerate(plots):
        plots_marker = _plot.count('</s>')
        story_marker = storys[idx].count('</s>')
        dic = {}
        if plots_marker != story_marker:
            count += 1
            # no_equal_list.append(idx)
            plot_list.append(plots_marker)
            story_list.append(story_marker)
            idx_list.append(idx + 1)

            # print('Line {} </s> markers number is\'nt equal, plot: {}, story: {}'.format((idx+1),plots_marker,story_marker))
        else:
            pass
    print('Non-equal count: {}, percentage: {} '.format(count, count/len(plots)))
    df = pd.DataFrame(list(zip(idx_list, plot_list, story_list)),
                      columns=['Line', '</s> in plot', '</s> in story'])
    df.to_csv(csvfile, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='type of train, string, test')
    args = parser.parse_args()

    plotfile = '/Users/yangjinrui/Documents/summer/storyGeneration/Plan-and-Write/data/writingPrompts/srl_resume/ready_train_model/plot.{}.txt'.format(args.type)
    storyfile = '/Users/yangjinrui/Documents/summer/storyGeneration/Plan-and-Write/data/writingPrompts/srl_resume/ready_train_model/story.{}.txt'.format(args.type)
    csvfile = '/Users/yangjinrui/Documents/summer/storyGeneration/Plan-and-Write/data/writingPrompts/srl_resume/ready_train_model/count_marker_no_equal.{}.csv'.format(args.type)
    # print('Check {}'.format(os.path.split(plotfile)[1].split('.')[-2]))
    print('Check {}'.format(args.type))
    check(plotfile,storyfile,csvfile)