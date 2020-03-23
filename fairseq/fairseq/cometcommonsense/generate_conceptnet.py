import os
import sys
import pickle
import json
import argparse

sys.path.append(os.getcwd()+'/cometcommonsense')
print(sys.path)
import torch

from fairseq.cometcommonsense.src.models import models as models
from fairseq.cometcommonsense.src.data import data as data
from fairseq.cometcommonsense.utils import utils as utils
from fairseq.cometcommonsense.src.data import config as cfg
from fairseq.cometcommonsense.src.interactive import functions as interactive

from fairseq.cometcommonsense.src.data.utils import TextEncoder

from tqdm import tqdm

from fairseq.cometcommonsense.src.evaluate.sampler import BeamSampler, GreedySampler, TopKSampler


def init():
    device = 0
    model_file = "/nas/home/tuhinc/fairseq/fairseq/cometcommonsense/pretrained_models/conceptnet_pretrained_model.pickle"
    sampling_algorithm = 'beam-5'
    model_stuff = data.load_checkpoint(model_file)
    opt = model_stuff["opt"]
    relations = data.conceptnet_data.conceptnet_relations
    if opt.data.get("maxr", None) is None:
        if opt.data.rel == "language":
            opt.data.maxr = 5
        else:
            opt.data.maxr = 1
    path = "/nas/home/tuhinc/fairseq/fairseq/cometcommonsense/data/conceptnet/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))
    data_loader = data.make_data_loader(opt)
    loaded = data_loader.load_data(path)

    encoder_path = "/nas/home/tuhinc/fairseq/fairseq/cometcommonsense/model/encoder_bpe_40000.json"
    bpe_path = "/nas/home/tuhinc/fairseq/fairseq/cometcommonsense/model/vocab_40000.bpe"

    text_encoder = TextEncoder(encoder_path, bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in relations]

    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    context_size_event = data_loader.max_e1
    context_size_effect = data_loader.max_e2

    n_special = len(special)
    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = models.make_model(
        opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False)

    models.load_state_dict(model, model_stuff["state_dict"])

    cfg.device = device
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
    model.eval()

    if "bs" in opt.eval:
        opt.eval.pop("bs")
    if "k" in opt.eval:
        opt.eval.pop("k")

    if "beam" in sampling_algorithm:
        opt.eval.sample = "beam"
        opt.eval.bs = int(sampling_algorithm.split("-")[1])
        sampler = BeamSampler(opt, data_loader)
    else:
        opt.eval.sample = "greedy"
        opt.eval.bs = 1
        sampler = GreedySampler(opt, data_loader)

    return model , sampler , data_loader, text_encoder

def generate(prefix,model,sampler,data_loader,text_encoder):
    e1 = prefix
    r = "Causes"
    sample_inputs = []
    output = interactive.get_conceptnet_sequence(
        e1, model, sampler, data_loader, text_encoder, r)
    return output
