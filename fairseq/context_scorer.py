import sys
import pickle

import torch 
import torch.autograd as autograd
#from .cnn_context_classifier import *

class ContextScorer():
    """build a scorer from a pretrained classifier"""
    def __init__(self, model_path, use_cuda=True):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f, map_location=lambda storage, loc: storage)
            if use_cuda:
                self.model.cuda()
            else:
                self.model.cpu()
        self.model.eval()
        self.normalize = torch.nn.LogSigmoid()
        self.use_cuda = use_cuda


    def int_to_tensor(self, tokens):
        if self.use_cuda:
            return autograd.Variable(torch.LongTensor(tokens)).cuda().view(-1, 1)
        else:
            return autograd.Variable(torch.LongTensor(tokens)).view(-1, 1)
    
    def __call__(self, init_tokens, conts, normalize_scores):
        self.model.eval()
        init_ind = self.int_to_tensor(init_tokens)
        conts_ind = [self.int_to_tensor(cont) for cont in conts]
        conts_ts = torch.cat(conts_ind, 1)   
        cont_lens = self.int_to_tensor([len(cont) for cont in conts])
        
        score_tensor = self.model(init_ind, (conts_ts, cont_lens))

        if normalize_scores:
            score_tensor = self.normalize(score_tensor.view(1, -1)).view(-1)
        else:
            score_tensor = score_tensor.view(-1)

        score = score_tensor.data.cpu().numpy()
        return score

