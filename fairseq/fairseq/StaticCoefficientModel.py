import sys

import torch
from torch import nn, optim
import numpy as np


class StaticCoefficientModel(nn.Module):
    def __init__(self, num_mods):
        super(StaticCoefficientModel, self).__init__()
        self.num_mods = num_mods
        self.coefs = nn.Linear(num_mods, 1, bias=False)
        self.coefs.weight.data = torch.FloatTensor(np.zeros((1, num_mods)))

    def forward(self, scores):
        #print(scores)
        return self.coefs(scores)


class CoefTrainer:
    """class for training scorer coefficients"""

    def __init__(self, num_scorers, ranking_loss, lr):
            self.weight_model = StaticCoefficientModel(num_scorers)
            self.use_ranking_loss = ranking_loss
            if self.use_ranking_loss:
                self.loss = nn.MarginRankingLoss()
            else:
                self.loss = nn.MSELoss()
            self.optimizer = optim.SGD(self.weight_model.parameters(), lr=lr)
            self.total_loss, self.total_n, self.total_correct = 0, 0, 0

    def train_coefficients(self, truth_lm_score, candidate_lm_score, gold_cont_raw_scores, candidate_raw_scores):
        self.weight_model.zero_grad()
        #truth_lm_scores = logprobs(model, [init_tokens + true_cont_tokens]).squeeze().cpu().data.numpy()  # this will be the shape of (len input x embed dimension) where len input is init + cont
        #truth_lm_score = sum([truth_lm_scores[i + len(init_tokens) - 1, true_cont_tokens[i]] for i in
        #                      range(len(true_cont_tokens))])  # this is just the probability of the sequence #TODO is it necessary for init and cont tokens to be separate?
        lm_scores = torch.Tensor([truth_lm_score, candidate_lm_score])  # this is the probability of the true sequence paired with the score of the best sequence. Both floats
        # print("LM pair", lm_scores)
        training_pair = [gold_cont_raw_scores, candidate_raw_scores]  # this is scorer scores of gold continuation, and of the best continuation. Both 1D arrays of len num scorers.
        training_pair = torch.Tensor(np.stack(training_pair))  # so this is now one row per scorer, with gold and best candidate as columns
        #print("Training pair", training_pair)
        # if self.use_cuda:
        #    training_pair.cuda()
        pair_scores = self.weight_model(training_pair).squeeze()
        #print("pair scores returned", pair_scores)
        pair_scores = pair_scores + lm_scores
        #print("pair scores concat", pair_scores)
        pred = pair_scores[0] - pair_scores[1]

        if self.use_ranking_loss:
            loss = self.loss((pair_scores[0]).unsqueeze(0),
                             (pair_scores[1]).unsqueeze(0), torch.ones(1))

        else:
            loss = self.loss(pred,
                             torch.FloatTensor([0]))  # use MSELoss, ((input-target)**2).mean()
        # print(loss.data.item())
        loss.requires_grad = True
        loss.backward()
        self.total_loss += loss.data.item()
        if self.use_ranking_loss and loss.data.item() == 0:
            self.total_correct += 1  # whether or not it is correct is whether the scorer did in fact say the gold was higher rank
        self.total_n += 1
        if self.total_n % 1 == 0:
            if self.use_ranking_loss:
                print('Train Accuracy: %f' % (self.total_correct / self.total_n))
            print('Loss: %f' % (self.total_loss / self.total_n))
            sys.stdout.flush()

        self.optimizer.step()
        self.weight_model.coefs.weight.data = self.weight_model.coefs.weight.data.clamp(min=0)

        return loss
