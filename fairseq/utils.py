import sys
#from importlib import import_module
import torch
#from cnn_context_classifier import CNNContextClassifier

def load_scorers(filepath):
    """expects a tsv of coefficients paired with scorer model info (used to load it in)"""
    scorer_config, coefs, model_info = [], [], []
    with open(filepath, "r") as scorer_file:
        for line in scorer_file:
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t') # expect coef and then name
            coefs.append(float(fields[0]))
            model_info.append(fields[1:])
            scorer_config.append(fields)
    print("Coefs:", coefs, file=sys.stderr)
    return coefs, model_info, scorer_config


# OLD VERSION
# def load_scorers(filepath, bart, cuda=False):
#     scorer_config, scorers, coefs = [], [], []
#     print("Creating scorers", file=sys.stderr)
#     with open(filepath) as scorer_file:
#         for line in scorer_file:
#             fields = line.strip().split('\t')
#             scorer_config.append(fields)
#             weight, module_path, classname = fields[:3]
#             weight = float(weight)
#             model_location = fields[3]
#             module = import_module(module_path)
#             constructor = getattr(module, classname)
#             model = CNNContextClassifier(300, 3, 0.5, bart)
#             x = torch.load(model_location)
#             model.load_state_dict(x)
#             scorer = constructor(model, cuda)
#             scorers.append(scorer)
#             coefs.append(weight)
#     print("Coefs:", coefs, file=sys.stderr)
#     return scorer_config, scorers, coefs
