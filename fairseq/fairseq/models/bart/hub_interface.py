# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from fairseq import utils
from fairseq.data import encoders
from fairseq.data.data_utils import collate_tokens
from fairseq.sequence_scorer import SequenceScorer


logger = logging.getLogger(__name__)


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False,
               gold_tokens: List[str] = None, **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        if gold_tokens:
            gold_tokens = [self.encode(tokens) for tokens in gold_tokens]
            max_l = kwargs.get("max_len_b", 512) # todo also don't make this hardcoded 512
            #breakpoint()
            if len(gold_tokens[0]) > max_l:
                gold_tokens = [torch.cat((gold_tokens[0][:max_l], gold_tokens[0][-1].unsqueeze(0)))] #TODO make this actually append the eos and also support more than 1
            
        hypos = self.generate(input, beam, verbose, gold_tokens=gold_tokens,
                              **kwargs)
        # for x in hypos:
        #     for i in range(0, len(x["tokens"]), 2):
        #         this = x["tokens"][i:i+2]
        #         print(this, self.decode(this))
        return [self.decode(x['tokens']) for x in hypos]

    def score_sequence(self, src_sents: List[str], tgt_sents: List[str], **kwargs):
        gen_args = copy.copy(self.args)
        setattr(gen_args, "score_reference", True)
        reference_scorer = self.task.build_generator(gen_args)

        src_sent_ids = [self.encode(sentence) for sentence in src_sents] # makes longtensors
        tgt_sent_ids = [self.encode(sentence) for sentence in tgt_sents]
        sample = self._build_sample(src_sent_ids)
        eos_idx, pad_idx = self.model.decoder.dictionary.eos_index, self.model.decoder.dictionary.pad_index
        # TODO don't hardcode cuda
        sample["net_input"]["prev_output_tokens"] = collate_tokens(tgt_sent_ids, pad_idx, eos_idx, move_eos_to_beginning=True).cuda()
        sample["target"] = collate_tokens(tgt_sent_ids, pad_idx, eos_idx).cuda() # if larger beam have to stack the tensor rather than just 0 index :p
        seq_score = reference_scorer.generate([self.model], sample)
        #lm_score = seq_score[0][0]["score"]
        return seq_score[0][0]["score"].data.item()

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)
        # for coefficient training need gold tokens
        gold_toks = kwargs.get("gold_tokens")
        gold_sample = self._build_sample(tokens) if gold_toks else None
        if gold_sample:
            eos_idx, pad_idx = self.model.decoder.dictionary.eos_index, self.model.decoder.dictionary.pad_index
            if self.args.cpu:
                shifted_gold = collate_tokens(gold_toks, pad_idx, eos_idx, move_eos_to_beginning=True)
                gold = collate_tokens(gold_toks, pad_idx, eos_idx)
            else:
                shifted_gold = collate_tokens(gold_toks, pad_idx, eos_idx, move_eos_to_beginning=True).cuda()
                gold = collate_tokens(gold_toks, pad_idx, eos_idx).cuda()

            gold_sample["net_input"]["prev_output_tokens"] = shifted_gold
            gold_sample["target"] = gold
            kwargs["gold_sample"] = gold_sample
            kwargs["gold_tokens"] = gold # TODO this is redundant now
        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(gen_args)
        if kwargs.get("learn"):
            setattr(gen_args, "score_reference", True) # so can get a sequence scorer
            ref_scorer = self.task.build_generator(gen_args)
            kwargs["reference_scorer"] = ref_scorer
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
            **kwargs
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
