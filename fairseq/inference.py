import torch
from fairseq.models.bart import BARTModel
import os
os.environ['CUDA_VISIBLE_DEVICES']="2, 3"

bart = BARTModel.from_pretrained(
    'checkpoint/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='temp'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 128
with open('./temp/val.source') as source, open('temp/val.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, temperature=0.7, lenpen=2.0, max_len_b=250, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n','') + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, sampling=True,  sampling_topk=5, temperature=0.7, lenpen=2.0, max_len_b=250, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n','') + '\n')
            fout.flush()
