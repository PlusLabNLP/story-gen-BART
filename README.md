# story-gen-BART


To finetune prompt to plot
==========================================================================================

Use the encoder.json and dict.txt already provided as a part of the repo, since it contains additional delimeter tokens relevant for story generations

 ```
  cd fairseq
  mkdir plot
 ```
 
 Since this is a seq2seq task you need source and target files
  - Put in plot directory 4 files train.source, train.target, val.source, val.target
  #sample data present
 
 Now for BPE preprocess:
  ```
    sh create_bpe.sh
  ```

Binarize dataset:

      ```
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --trainpref "plot/train.bpe" \
        --validpref "plot/val.bpe" \
        --destdir "plot/" \
        --workers 60 \
        --srcdict dict.txt \
        --tgtdict dict.txt
      ```

Download Pretrained BART from here:

https://github.com/pytorch/fairseq/tree/4b986c51ed89f9895df2f30e57909301b4a4f19b/examples/bart


Train models:

    ```
    sh run.sh
    ```

Update the field BART_PATH to suit where your pretained model.pt file is
You can customize  MAX_TOKENS and UPDATE_FREQ based on gpu memory / no of gpus

To finetune plot to story
==========================================================================================

Use the encoder.json and dict.txt already provided as a part of the repo, since it contains additional delimeter tokens relevant for story generations

 ```
  cd fairseq
  mkdir story
 ```
 
 Since this is a seq2seq task you need source and target files
  - Put in story directory 4 files train.source, train.target, val.source, val.target
   #sample data present


 Now for BPE preprocess:
  ```
    sh create_bpe.sh
  ```

Binarize dataset:

      ```
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --trainpref "story/train.bpe" \
        --validpref "story/val.bpe" \
        --destdir "story/" \
        --workers 60 \
        --srcdict dict.txt \
        --tgtdict dict.txt
      ```

Download Pretrained BART from here:

https://github.com/pytorch/fairseq/tree/4b986c51ed89f9895df2f30e57909301b4a4f19b/examples/bart


Train models:

    ```
    sh run.sh
    ```

Update the field BART_PATH to suit where your pretained model.pt file is
You can customize  MAX_TOKENS and UPDATE_FREQ based on gpu memory / no of gpus


Inference
================================================
For Inference:

  ```
    python inference.py
  ```
 
## Train Discriminators aka Classifier aka whatever we start calling them

To train a discriminator you will need to go through 3 steps:
1) generate positive and negative example data
2) compile tsv files for train and validation data
3) finetune the discriminator for us Roberta-large


