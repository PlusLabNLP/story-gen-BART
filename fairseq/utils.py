import sys
from importlib import import_module


def load_scorers(filepath, cuda=False):
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
            scorer = constructor(model_location, cuda)
            scorers.append(scorer)
            coefs.append(weight)
    print("Coefs:", coefs, file=sys.stderr)
    return scorer_config, scorers, coefs