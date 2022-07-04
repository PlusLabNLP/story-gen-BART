'''
A script that
Command to run:
'''

import sys
import re

def recover_hyperparams(filename):
    """hacky. gets the hyperparams for RAKE from the filename to use in generating new filenames"""
    h_params = re.findall('\d', filename)
    return h_params

if __name__ == "__main__":
    title_file, kw_file, story_file = sys.argv[1], sys.argv[2], sys.argv[3]
    data_type = sys.argv[4] # train, dev, test
    target_dir = sys.argv[5]
    titles, keywords, stories = [],[],[]
    title_kw_pattern = '{0} <EOT> {1} <EOL>'
    kw_story_pattern = '{0} <EOL> {1}'
    all_pattern = '{0} <EOT> {1} <EOL> {2}'
    base_filename = 'WP'
    #h_params = '.'.join(recover_hyperparams(kw_file))


    with open(title_file, 'r') as infile:
        for line in infile:
            titles.append(line.strip())
    with open(kw_file, 'r') as infile:
        for line in infile:
            keywords.append(line.strip())
    with open(story_file, 'r') as infile:
        for line in infile:
            stories.append(line.strip())
    total_lines = len(titles)
    title_key_file = '{2}/{0}.titlesepkey.{1}'.format(base_filename, data_type,
                                                          target_dir)
    title_key_story_file = '{2}/{0}.titlesepkeysepstory.{1}'.format(base_filename, data_type,
                                                                        target_dir)
    key_story_file = '{2}/{0}.keysepstory.{1}'.format(base_filename, data_type, target_dir)

    with open(title_key_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(title_kw_pattern.format(titles[i], keywords[i]))
            outfile.write('\n')

    with open(title_key_story_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(all_pattern.format(titles[i], keywords[i], stories[i]))
            outfile.write('\n')

    with open(key_story_file, 'w') as outfile:
        for i in range(total_lines):
            outfile.write(kw_story_pattern.format(keywords[i], stories[i]))
            outfile.write('\n')


