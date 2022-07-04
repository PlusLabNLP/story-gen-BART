'''
A script that
Command to run:
'''

import sys
import re
import argparse



def recover_hyperparams(filename):
    """hacky. gets the hyperparams for RAKE from the filename to use in generating new filenames"""
    h_params = re.findall('\d', filename)
    return h_params

def title_key(title_file, kw_file, title_key_file):
    titles, keywords= [], []
    title_kw_pattern = '{0} <EOT> {1} <EOL>'
    with open(title_file, 'r') as infile:
        for line in infile:
            titles.append(line.strip())
    with open(kw_file, 'r') as infile:
        for line in infile:
            keywords.append(line.strip())
    print("Concat {0} title with {1} storylines".format(len(titles), len(keywords)))
    total_lines = len(titles)
    with open(title_key_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(title_kw_pattern.format(titles[i], keywords[i]))
            outfile.write('\n')

def title_key_story(title_file, kw_file, story_file, title_key_story_file):
    titles, keywords, stories = [], [], []
    all_pattern = '{0} <EOT> {1} <EOL> {2}'
    with open(title_file, 'r') as infile:
        for line in infile:
            titles.append(line.strip())
    with open(kw_file, 'r') as infile:
        for line in infile:
            keywords.append(line.strip())
    with open(story_file, 'r') as infile:
        for line in infile:
            stories.append(line.strip())
    print("Concat {0} title with {1} storylines and {2} stories".format(len(titles), len(keywords), len(stories)))
    total_lines = len(titles)
    with open(title_key_story_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(all_pattern.format(titles[i], keywords[i], stories[i]))
            outfile.write('\n')

def key_story(kw_file, story_file, key_story_file):
    keywords, stories = [],[]
    kw_story_pattern = '{0} <EOL> {1}'
    with open(kw_file, 'r') as infile:
        for line in infile:
            keywords.append(line.strip())
    with open(story_file, 'r') as infile:
        for line in infile:
            stories.append(line.strip())
    print("Concat {0} storylines with {1} stories".format(len(keywords), len(stories)))
    total_lines = len(keywords)
    with open(key_story_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(kw_story_pattern.format(keywords[i], stories[i]))
            outfile.write('\n')

def title_story(title_file, story_file, title_story_file):
    titles, stories = [],[]
    title_story_pattern = '{0} <EOT> {1}'
    with open(title_file, 'r') as infile:
        for line in infile:
            titles.append(line.strip())
    with open(story_file, 'r') as infile:
        for line in infile:
            stories.append(line.strip())
    print("Concat {0} titles with {1} stories".format(len(titles), len(stories)))
    total_lines = len(titles)
    with open(title_story_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(title_story_pattern.format(titles[i], stories[i]))
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--title_file', type=str, help='path to input file')
    parser.add_argument('--kw_file', type=str, help='path to storyline file')
    parser.add_argument('--story_file', type=str, help='path to story file')
    parser.add_argument('--target_dir', type=str, help='path to save output file')
    parser.add_argument('--data_type', type=str, help='data type')
    parser.add_argument('--title_key', action='store_true', help='concat title and storyline')
    parser.add_argument('--title_key_story', action='store_true', help='concat title, storyline and story')
    parser.add_argument('--key_story', action='store_true', help='concat storyline and story')
    parser.add_argument('--title_story', action='store_true', help='concat title and story')
    args = parser.parse_args()
    title_file = args.title_file
    kw_file = args.kw_file
    story_file = args.story_file
    data_type = args.data_type # train, dev, test
    target_dir = args.target_dir
    base_filename = 'WP'
    #h_params = '.'.join(recover_hyperparams(kw_file))

    if args.title_key:
        title_key_file = '{2}/{0}.titlesepkey.{1}'.format(base_filename, data_type,
                                                          target_dir)
        title_key(title_file, kw_file, title_key_file)

    if args.title_key_story:
        title_key_story_file = '{2}/{0}.titlesepkeysepstory.{1}'.format(base_filename, data_type,
                                                                        target_dir)
        title_key_story(title_file, kw_file, story_file, title_key_story_file)

    if args.key_story:
        key_story_file = '{2}/{0}.keysepstory.{1}'.format(base_filename, data_type, target_dir)
        key_story(kw_file, story_file, key_story_file)

    if args.title_story:
        title_story_file = '{2}/{0}.titlesepstory.{1}'.format(base_filename, data_type, target_dir)
        title_story(title_file, story_file, title_story_file)



