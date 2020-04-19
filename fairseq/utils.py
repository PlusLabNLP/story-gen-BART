import sys
from importlib import import_module
import torch
from cnn_context_classifier import CNNContextClassifier

def load_scorers(filepath, bart, cuda=False):
    scorer_config, scorers, coefs = [], [], []
    print("Creating scorers", file=sys.stderr)
    with open(filepath) as scorer_file:
        for line in scorer_file:
            fields = line.strip().split('\t')
            scorer_config.append(fields)
            weight, module_path, classname = fields[:3]
            weight = float(weight)
            model_location = fields[3]
            module = import_module(module_path)
            constructor = getattr(module, classname)
            model = CNNContextClassifier(300, 3, 0.5, bart)
            x = torch.load(model_location)
            model.load_state_dict(x)
            scorer = constructor(model, cuda)
            scorers.append(scorer)
            coefs.append(weight)
    print("Coefs:", coefs, file=sys.stderr)
    return scorer_config, scorers, coefs
