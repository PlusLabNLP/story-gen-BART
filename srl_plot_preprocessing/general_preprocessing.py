'''
A script that preprocesses any text file based on given criteria
- can replace tokens below a given threshold with unk tokens (or other tokens)
- can remove unwanted chars
- can lowercase
- can truncate
- does NER recognition as part of the vocab creation when thresholding (this is default, not controlled by a flag, it will only not do this if not using thresholding)

Writes versions of the original files back to the dir after preprocessing
Command to run:

python preprocessing/general_preprocessing.py data/WritingPrompts/test --truncate --threshold 10 - --lower --sent-tok punkt
'''

import argparse
from collections import Counter, defaultdict
import os
import sys
import re
import spacy
import nltk
from nltk import sent_tokenize


class NEREntity:
    """a class for doing NER preprocessing"""
    def __init__(self, ent_string, ent_type, id):
        self.ent_string = ent_string
        self.ent_type = ent_type
        self.id = id
        self.aka = set(ent_string.split('_')) if '_' in ent_string else set()

    def __str__(self):
        """Entities will print as the type and the locally unique ID, e.g. <PERSON+1>"""
        return "<{}+{}>".format(self.ent_type, self.id)


def get_vocab(files):
    """takes a list of files, returns a counter of frequencies of all tokens"""
    all_vocab = Counter()
    for file in files:
        with open(file, "r") as infile:
            vocab_counter = Counter(infile.read().strip().split())
        all_vocab += vocab_counter
    return all_vocab


def find_below_threshold(vocab_dict, threshold):
    """takes a counter of vocab items and an int and returns a set of words below threshold"""
    return {k for k in vocab_dict if vocab_dict[k] < threshold}


def replace_all_with_char(replace_set, replace_dict, files, has_whitespace=False):
    """takes a set of things to be replaced, a dict of things to replace them with, files, and a boolean for whether the character to be replaced contains internal whitespace.
    No return, modifies files in place."""

    print("Starting replacing {} in {} files".format(replace_set, len(files)), file=sys.stderr)
    for file in files:
        with open(file, "r") as infile:
            all_lines = infile.readlines()
        with open(file, "w") as outfile:
            for line in all_lines:
                if has_whitespace: # have to use regex even though slower
                    outline = line
                    for pattern in replace_set:
                        replace_str = replace_dict.get(pattern, '<unk>')
                        outline = re.sub(pattern, replace_str, outline)
                    outline = outline.strip()
                else:
                    newline = []
                    for tok in line.strip().split():
                        if tok in replace_set:
                            if '_' in tok: #this is special cases to split phrases before making them unks
                                print("Splitting {}".format(tok))
                                new_toks = tok.split('_')
                                for i in range(len(new_toks)):
                                    if new_toks[i] in replace_set:
                                        new_toks[i] = replace_dict.get(new_toks[i], '<unk>')
                                tok = " ".join(new_toks)
                            else:
                                tok = replace_dict.get(tok, '<unk>')
                            newline.append(tok)
                        else:
                            newline.append(tok)
                    outline = " ".join(newline)

                outfile.write("{}\n".format(outline))
    print("Done\n", file=sys.stderr)


def truncate(files, max_tok, start_char=None, end_char=None):
    """
    truncates text in files, writes over input files.
    If start_char is None, assumes start at the beginning of the line.
    If end_char is None, assumes each line is a block to be truncated
    No return -> modifies files in place
    """

    print("Starting truncating {} files".format(len(files)), file=sys.stderr)
    for file in files:
        outlines = []
        with open(file, "r") as infile:
            for line in infile:
                if start_char:
                    start, rest = line.strip().split(start_char, 1)
                    trunc_line = "{} {} {}".format(start.strip(), start_char,
                                                   ' '.join(rest.strip().split()[:max_tok]))
                else:
                    trunc_line = ' '.join(line.strip().split()[:max_tok])
                outlines.append(trunc_line)
        with open(file, "w") as outfile:
            outfile.write("\n".join(outlines))
    print("Done\n", file=sys.stderr)


def to_lower(files, special_chars):
    """lowercases files, excepting a set of special characters. Modifies files in place""" 
    print("Lowercasing {} files\nIgnoring: {}".format(len(files), special_chars), file=sys.stderr)
    for file in files:
        outdata = []
        with open(file, "r") as infile:
            for line in infile:
                outline = " ".join([tok if tok in special_chars else tok.lower()
                                    for tok in line.strip().split()])
                outdata.append(outline)
        with open(file, "w") as outfile:
            outfile.write("\n".join(outdata))
    print("Done\n", file=sys.stderr)


def write_sentence_sep(files, sep_char, sent_detector, chop=False, tok_type=None, start_char=None):
    """
    Tokenizes a list of file paths into sentences using a sent_detector, and delimits them with a given sep_char.
    Supports either a spacy or an NLTK punkt model object as sent_detector, and the type is declared in tok_type.
    if chop is True then the final sentence is removed - this handles a case where it may have been truncated.
    if start_char is given, things before start_char are not modified.
    Returns None -> modifies in place.
    """
    print("Tokenizing Sentences in {} files and adding separation char: {}".format(len(files), sep_char), file=sys.stderr)
    for file in files:
        outdata = []
        with open(file, "r") as infile:
            for line in infile:
                if start_char:
                    prefix, story = line.strip().split(start_char)  # this is to be sure we don't mess with non-story parts.
                else:
                    story = line.strip()
                detok_story = re.sub('\s(?=[.])', '', story)
                if tok_type == "spacy":
                    story_sentences = [sent.text for sent in sent_detector(detok_story).sents]
                elif tok_type == "punkt":
                    story_sentences = sent_detector(detok_story) #TODO make sure the else (of the if-else) is taken care of
                if chop:
                    story_sentences.pop()  # This removes the final sentence, which may be incomplete
                # re-tokenize. Not using something like word_tokenize because it messed with special chars
                tok_story_sentences = [re.sub('(?<=\S)[.]', ' .', sent) for sent in story_sentences]

                outline = " {} ".format(sep_char).join(tok_story_sentences)
                if start_char:
                    outline = prefix + " {} {} ".format(start_char, sep_char) + outline
                outdata.append(outline)
        with open(file, "w") as outfile:
            outfile.write("\n".join(outdata))
    print("Done\n", file=sys.stderr)


def ner_processing(nlp_model, files, target_dir="./"):
    """takes a Spacy model capable of NER, a set of files, and a target_dir for output. 
    Modifies files in place, but also writes out the entity to string mappings used. 
    returns the strings used as entities to they can be tracked as special chars.
    Not very fast."""

    print("Running Named Entity Recognition on {} files".format(len(files)), file=sys.stderr)

    all_entities, key_val_pairs = set(), []
    non_entity_caps = set() # used to store things that might have been missed by NER. Will get all things that are start of sentence caps, but those should all be pretty common so won't be picked up by thresholding anyway.
    #id2entity, id2string = [None], [args.unk]
    internal_whitespace = re.compile('(?!<=^)\s+(?!=$)')

    for file in files:
        all_stories = []
        with open(file, "r") as infile:
            for line in infile:
                # string to entity stores an entity based on a string, entity ids is used to set ids by entity type
                string2entity, entity_ids, multi_word_ents = {}, Counter(), set()

                nlp_text = nlp_model(line)
                # find entities, assign entities with ids and make mappings, removing whitespace
                for entity in nlp_text.ents:
                    ent_string = internal_whitespace.sub('_', entity.text).lower()
                    # keep multi_word ones for later merging
                    if '_' in ent_string:
                        multi_word_ents.add(ent_string)

                    if ent_string not in string2entity:
                        # make a mapping where entity ids are local to each story. As stories are one per line
                        string2entity[ent_string] = NEREntity(ent_string, entity.label_, entity_ids[entity.label_])
                        entity_ids[entity.label_] += 1 # increment the id for each type

                    entity.merge() # this merges entity tokens into one token if it spans multiple

                # TODO break out into sep function
                # if entities exist in multiword settings assume same entity and merge ids. This is to handle stuff like John Smith
                has_title = set() # this is to handle things like Mr. and Miss separately
                titles = {'mr', 'mrs', 'miss', 'sir', 'lady', 'lord', 'ms', 'dr', 'doctor',
                             'general', 'captain', 'father', 'count', 'countess', 'baron',
                             'baroness', 'king', 'queen', 'prince', 'princess', 'madam', 'earl'}

                # Clean up the mappings - this is basically cause NER isn't good enough and sometimes catches just a title
                for title in (titles & set(string2entity)):
                    del string2entity[title]

                for ent_str in multi_word_ents:
                    ent_tokens = ent_str.split('_')
                    if ent_tokens[0] in titles:
                        has_title.add(ent_str)
                        continue
                    for tok in ent_tokens:
                        if tok in string2entity:
                            if string2entity[tok].ent_type == string2entity[ent_str].ent_type:
                                string2entity[tok].id = string2entity[ent_str].id
                for ent_str in has_title:
                    ent_tokens = ent_str.split('_')
                    for i in range(1, len(ent_tokens)):
                        tok = ent_tokens[i]
                        if tok in string2entity:
                            if string2entity[tok].ent_type == string2entity[ent_str].ent_type:
                                string2entity[tok].id = string2entity[ent_str].id

                # Heuristic, if a string is not an entity but is capitalized it might be an entity. The particularly effects entities that appear only at the start of sentences.
                non_entity_caps.update(set([internal_whitespace.sub('_', tok.text).lower() for tok in nlp_text
                                   if (tok.ent_type == 0 and tok.text.istitle())]))

                # make substitutions
                full_story = []
                for tok in nlp_text:
                    new_tok = internal_whitespace.sub('_', tok.text)
                    if new_tok.lower() in string2entity:
                        new_tok = str(string2entity[new_tok.lower()])
                    full_story.append(new_tok)
                #full_story = [internal_whitespace.sub('_', tok.text) for tok in nlp_text] # can add if tok.ent_type > 1 if want to only do this for entities
                all_stories.append(" ".join(full_story))

                # store all entity replacements for later special characters
                all_entities.update(set([str(val) for val in string2entity.values()]))
                # and for writing out
                key_val_pairs.extend(["{} {}".format(key, str(val)) for key, val in
                                 sorted(string2entity.items())])

        with open(file, "w") as outfile:
            outfile.write("\n".join(all_stories))
            print("Finished 1 file", file=sys.stderr)
        print("Done", file=sys.stderr)

    # Print string 2 entity file
    string2entity_file = "string2entity.txt"
    print("Writing {} to target_directory (defaults are this one and inputdir)".format(string2entity_file), file=sys.stderr)
    with open(target_dir+string2entity_file, "w") as outfile:
        #key_val_pairs = ["{} {}".format(key, str(val)) for key, val in sorted(string2entity.items())]
        outfile.write("\n".join(key_val_pairs))

    return all_entities, non_entity_caps



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('inputdir', type=str, help='dir of files to modify. Need all at once to get accurate vocab')
    p.add_argument('--unk', type=str, default='<unk>',
                   help='the string to use as an unk token')
    p.add_argument('--threshold', type=int, default='1',
                   help='the threshold of frequency below which tokens are replaces with unks. Set to 1 to prevent thresholding')
    p.add_argument('--ner', action='store_true', help="whether to run NER on input")
    p.add_argument('--truncate', action='store_true', help="whether to truncate stories")
    p.add_argument('--lower', action='store_true', help="whether to lowercase stories")
    p.add_argument('--sent-tok', type=str, default=None, help="whether to tokenize stories into "
                                                              "sentences. If not none, spacy or punkt")
    p.add_argument('--sep', type=str, default='</s>', help="the sentence delimeter to use with sent-tok")
    p.add_argument('--max-tok', type=int, default=1000, help="max tok per story, used if truncate is True")
    p.add_argument('--remove', type=str, default='', help="tokens or char to remove, pipe-delimited. Case-sensitive even if lowercasing")
    p.add_argument('--start-char', type=str, default=None, help="char marking start of text to be truncated, if truncate is True")
    p.add_argument('--has-whitespace', action='store_true', help="whether the characters to remove have internal whitespace")
    args = p.parse_args()

    special_chars = {'<EOL>', '<EOT>', '<eos>', '</s>', '#', '<P>', args.unk, args.sep} # harcoded list of special characters not to touch

    nlp = None  # this is for spacy use but it is slow so we don't load it if not necessary
    spacy_model = 'en_core_web_lg'

    with os.scandir(args.inputdir) as source_dir:
        files = sorted([file.path for file in source_dir if file.is_file() and not file.name.startswith('.')])
        #vocab_files  = [file.path for file in source_dir if file.is_file() and not file.name.startswith('.')]
        #ner_files = [file.path for file in source_dir if file.is_file() and file.name.startswith('WP.story') and not file.name.endswith('dev')]

    if args.remove:  # remove before truncation so not counted
        remove_chars = set(args.remove.split('|'))
        assert(remove_chars & special_chars == set()), "Cannot have a special character as a character to remove"
        # note that this will only work if the replace characters are not whitespace separated
        replace_all_with_char(remove_chars, '', files, has_whitespace=args.has_whitespace)

    if args.truncate:
        #truncate first so shorter for later processing
        truncate(files, args.max_tok, args.start_char)

    # get entities like names that shouldn't be removed BEFORE lowercasing
    if args.ner:
        nlp = spacy.load(spacy_model)

        # Need to special case all special chars for tokenization
        for key in special_chars:
            nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])

        entity_strings, non_entity_caps = ner_processing(nlp, files, args.inputdir)
        special_chars.update(entity_strings) #entity strings are now special chars

    if args.lower:
        to_lower(files, special_chars)

    # threshold AFTER lowercasing
    if args.threshold > 1:
        vocab = get_vocab(files)
        below_threshold = find_below_threshold(vocab, args.threshold)
        # make sure that special characters are not in the remove list and
        remove_list = below_threshold - special_chars
        #print("The below were removed from the remove list since they were previously "
              #"capitalized:\n{}".format(below_threshold & non_entity_caps), file=sys.stderr)
        replace_all_with_char(remove_list, {}, files)

    #tokenize last
    if args.sent_tok:
        print("using {} as a sentence tokenizer".format(args.sent_tok), file=sys.stderr)
        if args.sent_tok == 'punkt':
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Installing punkt to active environment", file=sys.stderr)
                nltk.download('punkt')
            sent_detector = sent_tokenize
        elif args.sent_tok == 'spacy':
            if not nlp:
                nlp = spacy.load(spacy_model)
                for key in special_chars:
                    nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])
            sent_detector = nlp
        else:
            sys.exit("unsupported sentence tokenizer")
        write_sentence_sep(files, args.sep, sent_detector, chop=args.truncate, tok_type=args.sent_tok, start_char=args.start_char)








