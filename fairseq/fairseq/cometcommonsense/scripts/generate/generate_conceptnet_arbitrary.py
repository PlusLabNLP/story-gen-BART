import os
import sys
import pickle
import json
import argparse

sys.path.append(os.getcwd()+'/comet-commonsense')
print(sys.path)
import torch

import src.models.models as models
import src.data.data as data
import utils.utils as utils
import src.data.config as cfg
import src.interactive.functions as interactive

from src.data.utils import TextEncoder

from tqdm import tqdm

from src.evaluate.sampler import BeamSampler, GreedySampler, TopKSampler



parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")
parser.add_argument("--output_file", type=str, default="tmp/output.json")
parser.add_argument("--input", type=str, default="")
parser.add_argument("--sampling_algorithm", type=str, default="greedy")

args = parser.parse_args()

model_stuff = data.load_checkpoint(args.model_file)
opt = model_stuff["opt"]

relations = data.conceptnet_data.conceptnet_relations

if opt.data.get("maxr", None) is None:
    if opt.data.rel == "language":
        opt.data.maxr = 5
    else:
        opt.data.maxr = 1

path = "comet-commonsense/data/conceptnet/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))
data_loader = data.make_data_loader(opt)
loaded = data_loader.load_data(path)

encoder_path = "comet-commonsense/model/encoder_bpe_40000.json"
bpe_path = "comet-commonsense/model/vocab_40000.bpe"

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

cfg.device = args.device
cfg.do_gpu = True
torch.cuda.set_device(cfg.device)
model.cuda(cfg.device)

model.eval()

# Format
# {"e1": seed entity, "r": desired relation (i.e., PartOf)}
# Capitalization must be maintained
print(args.input)
sample_inputs = []
val = {"e1":args.input, "r": "Causes"}
sample_inputs.append(val)

if "bs" in opt.eval:
    opt.eval.pop("bs")
if "k" in opt.eval:
    opt.eval.pop("k")

if "beam" in args.sampling_algorithm:
    opt.eval.sample = "beam"
    opt.eval.bs = int(args.sampling_algorithm.split("-")[1])
    sampler = BeamSampler(opt, data_loader)
else:
    opt.eval.sample = "greedy"
    opt.eval.bs = 1
    sampler = GreedySampler(opt, data_loader)

outputs = []


for input_case in tqdm(sample_inputs):
    e1 = input_case["e1"]
    r = input_case["r"]

    output = interactive.get_conceptnet_sequence(
        e1, model, sampler, data_loader, text_encoder, r)

    outputs.append(output)

json.dump(outputs, open(args.output_file, "w"))
