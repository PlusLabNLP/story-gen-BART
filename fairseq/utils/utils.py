import json
import copy

import torch

import numpy as np
import contextlib
import nltk
from distutils.dir_util import mkpath
from urllib.parse import quote
from tqdm import tqdm
from pattern.en import conjugate, lemma, lexeme,PRESENT,PAST,PARTICIPLE,SG
m = {}
import os
words = []
print(os.getcwd())

def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None,
              do_epoch=True):
    string = prefix
    string += "{}-{}".format(opt.dataset, opt.exp)
    string += "/"
    string += "{}-{}-{}".format(opt.trainer, opt.cycle, opt.iters)
    string += "/"
    string += opt.model
    if opt.mle:
        string += "-{}".format(opt.mle)
    string += "/"
    string += make_name_string(opt.data) + "/"

    string += make_name_string(opt.net) + "/"
    string += make_name_string(opt.train.static) + "/"

    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(
        opt.train.dynamic, True, do_epoch, set_epoch)
    if is_dir:
        mkpath(string)

    return string


def make_name_string(dict_, final=False, do_epoch=False, set_epoch=None):
    if final:
        if not do_epoch:
            string = "{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs)
        elif set_epoch is not None:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.items():
        if type(v) == DD:
            continue
        if isinstance(v, list):
            val = "#".join(is_bool(str(vv)) for vv in v)
        else:
            val = is_bool(v)
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string

def getAllConcepts():
    for line in open(os.getcwd()+'/data/concept.txt'):
        words.append(line.strip())
    return words


def updateConcept(concept):
    f = open(os.getcwd()+'/data/concept.txt','a')
    f.write(concept+'\n')


def filterSentences(keyword, sentences):
    s = []
    for sent in sentences:
        sent = sent.lower()
        if sent.startswith(keyword) or sent.endswith(keyword) or sent.endswith(keyword+'.') or sent.endswith(keyword+'?'):
            s.append(sent.capitalize())
    return s


def isPageInValid(f):
    elem = f.split('<div id="all">')
    if len(elem)==1:
        return True
    return False

def getLast(f):
    if '>last<span style=' in f:
        elem = f.split('>last<span style=')
        last = int(elem[0].split('a href="')[-1].split('_')[1].replace('.html"',''))
        return last
    else:
        return 1

def getSentencesOnline(keyword):
    f1 = open(os.getcwd()+'/data/corpus.txt','a')
    flag = True
    if keyword in m:
        flag = False
    keyword = quote(keyword)
    originalurl = '"https://sentencedict.com/'+keyword+'.html"'
    url = originalurl
    command = 'wget '+originalurl+' -q -nv -O comet-commonsense/temp/sent.txt'
    os.system(command)
    f = open(os.getcwd()+'/comet-commonsense/temp/sent.txt','r',encoding ='ISO-8859-1').read()
    c = 1
    if isPageInValid(f):
        return []
    else:
        s = []
        while True:
            if c>1:
                elem = f.split('<div id="all">')
                sentences = elem[1].split('</div></div>')[0]
                sentences = sentences.split('</div><div>')
                for line in sentences:
                    line = line.replace('<em>','')
                    line = line.replace('</em>','')
                    if len(line.split(', '))==2:
                        line = line.split(', ')[1]
                        if '</div>' in line:
                            continue
                        s.append(line)
                    elif len(line.split('. '))==2:
                        line = line.split('. ')[1]
                        if '</div>' in line:
                            continue
                        s.append(line)
                url = originalurl.replace(keyword,keyword+'_'+str(c))
                os.system('wget '+url+' -q -nv -O comet-commonsense/temp/sent'+str(c)+'.txt')
                f = open('comet-commonsense/temp/sent'+str(c)+'.txt','r',encoding ='ISO-8859-1').read()
                if c>getLast(f):
                    break
            c = c+1
        if flag:
            for line in s:
                if 'Sentencedict' in line or 'nbsp' in line or 'href' in line:
                    continue
                f1.write(line+'\n')
        return s


def getSentences(keyword):
    concepts = getAllConcepts()
    if keyword in concepts:
        retrieval_corpus = open(os.getcwd()+'/data/corpus.txt')
        sentences = [sent.strip() for sent in retrieval_corpus.readlines()]
        s = filterSentences(keyword,sentences)
        if len(s)>0:
            return s
    else:
        sentences = getSentencesOnline(keyword)
        if len(sentences)>0:
            updateConcept(keyword)
        return filterSentences(keyword,sentences)


def filter_beam_output(arr, inp):
    print("Beam is ",arr)
    words = inp.split()
    stop_phrase = ['person to be','person to get','person will','you will','you to'
    'her to be','you have','get into','you to be','they get','have','you to get']
    flag = True
    res = ''
    additional = ''
    for i in range(0, 5):
        beam_w = arr[i].split()
        tags = nltk.pos_tag(nltk.word_tokenize(arr[i]))
        if len(set(beam_w).intersection(set(words))) > 0 or ('allerg' in arr[i] and 'allerg' in inp ) or ('sink' in arr[i]):
            continue
        if len(beam_w)==1:
            try:
                x = conjugate(verb=arr[i],tense=PARTICIPLE,number=SG)
            except:
                x = arr[i]
        if len(beam_w)==1 and x in inp:
            continue
        for phrase in stop_phrase:
            if phrase in arr[i]:
                arr[i] = arr[i].replace(phrase+' ','')
        if len(beam_w)==3 and 'person to ' in arr[i]:
            arr[i] = arr[i].replace('person to ','')
        if 'you get ' in arr[i]:
            arr[i] = arr[i].replace('you get ','get ')
        if ' for everyone' in arr[i]:
            arr[i] = arr[i].replace(' for everyone','')
        if arr[i].startswith('get ') and arr[i]!='get up' and arr[i]!='get sick' :
            arr[i] = arr[i].replace('get ','')
        if len(beam_w)==2 and 'feel' in arr[i] and 'bad' not in arr[i]:
            arr[i] = arr[i].replace('feel ','')
            arr[i] = arr[i].replace(' feel ','')
            arr[i] = arr[i].replace(' feel','')
        if len(beam_w)==2 and arr[i].startswith('fail'):
            additional = arr[i].replace('fail ','')
            arr[i] = 'fail'
        if arr[i].startswith('be '):
            arr[i] = arr[i].replace('be ','')
        if arr[i]=='you eat it':
            arr[i]='eat'
        if arr[i]=='global thermonuclear war' or arr[i]=='masturbate' or arr[i]=='pessimistic':
            continue
        if arr[i].startswith('you ') and len(beam_w)>=3:
            arr[i] = arr[i].replace('you ','')
        if len(arr[i].split())>1 and arr[i].split()[1]=='from':
            arr[i] = arr[i].split()[0]
        if len(getSentences(arr[i]))>3:
            return arr[i],additional
        if len(arr[i].split())>2 and len(tags)==3 and (tags[2][1] =='NN') and (tags[1][1] !='JJ'):
            additional = arr[i].replace(tags[2][0],'')
            arr[i] = tags[2][0]
        if len(arr[i].split())>2 and len(tags)==3 and (tags[2][1] =='NN') and (tags[1][1] =='JJ'):
            arr[i] = tags[1][0]+' '+tags[2][0]
        if len(arr[i].split())>=3 and len(tags)==4 and (tags[3][1] =='NN'):
            arr[i] = tags[3][0]
        if len(arr[i].split())>2 and len(tags)==3 and (tags[0][1] =='NN') and (tags[2][1] =='VB'):
            additional = arr[i].replace(tags[0][0],'')
            arr[i] = tags[0][0]
        if len(arr[i].split())>4 and len(tags)==5 and (tags[0][1]=='NN'):
            additional = arr[i].replace(tags[0][0],'')
            arr[i] = tags[0][0]
        if arr[i]=='tire' or arr[i]=='hospitalize' or arr[i]=='bore': #make it modular to convert to past tense
            arr[i] = arr[i]+'d'
        if len(getSentences(arr[i]))>3:
            return arr[i],additional
        if len(arr[i].split())==2 and len(tags)==2 and (tags[0][1]=='JJ') and (tags[1][1]=='NN') and additional=='':
            additional = arr[i].replace(tags[1][0],'')
            arr[i] = tags[1][0]
        if len(getSentences(arr[i]))>3:
            return arr[i],additional
        print("Arr[i] for past is",arr[i])
        print(conjugate(verb=arr[i],tense=PAST,number=SG))
        if len(getSentences(conjugate(verb=arr[i],tense=PAST,number=SG)))>3:
            return conjugate(verb=arr[i],tense=PAST,number=SG),additional

def is_bool(v):
    if str(v) == "False":
        return "F"
    elif str(v) == "True":
        return "T"
    return v


def generate_config_files(type_, key, name="base", eval_mode=False):
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    # for param in changes[key]:
    #     base_config[param] = changes[key][param]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


def initialize_progress_bar(data_loader_list):
    num_examples = sum([len(tensor) for tensor in
                        data_loader_list.values()])
    return set_progress_bar(num_examples)


def set_progress_bar(num_examples):
    bar = tqdm(total=num_examples)
    bar.update(0)
    return bar


def merge_list_of_dicts(L):
    result = {}
    for d in L:
        result.update(d)
    return result


def return_iterator_by_type(data_type):
    if isinstance(data_type, dict):
        iterator = data_type.items()
    else:
        iterator = enumerate(data_type)
    return iterator


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def flatten(outer):
    return [el for inner in outer for el in inner]


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


# Taken from Jobman 0.1
class DD(dict):
    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
#         if attr.startswith('__'):
#             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z
