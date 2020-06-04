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
  #sample data present in fairseq/plot
 
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
  #sample data present in fairseq/story


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
 
Train Aristotelian Rescorers (aka classifiers, aka discriminators)
================================================

To train a rescorer you will need to go through 3 steps:
1) split the prompt + plot data into source and target
2) generate positive and negative example data
3) compile tsv files for train and validation data
4) finetune the discriminator for us Roberta-large

For 1-3):
Use the script `preprocessing/make_cc_version_pnw_data.py` to split data into source and targets for creating training data.
Then use the script `preprocessing/create_classifier_dataset.py` to generate positive and negative examples and the tsv.


Preprocess task data and it needs binary .bin files to finetune roberta:
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>

We use it as a proxy for RTE task (sentence pair with label 0 or 1) so recommend to use it

Fine-tuning on GLUE task:

Use run_disc.sh


Finetune 4 discriminators

     1.0 /nas/home/fairseq/relevance-roberta/  checkpoint_best.pt  roberta.large/
     1.0 /nas/home/fairseq/eventinter-roberta  checkpoint_best.pt  roberta.large/
     1.0 /nas/home/fairseq/eventintraV-roberta checkpoint_best.pt  roberta.large/
     1.0 /nas/home/fairseq/entity-roberta  checkpoint_best.pt  roberta.large/


Mixture Weight training
================================================

Next step is mixture coefficient training
train_coefs.py is a decoding script (like inference.py) that specifically trains coefficients. 

The rescoring is done here:

    ./story-gen-BART/fairseq/fairseq/search.py#L267-L324

This is the method that sequence generator calls in order to sample the next k hypotheses and then return them as candidates along with their probabilities.

In these 3 lines we concat the source tokens and all tokens generated so far with the current k hypotheses: 

    ./story-gen-BART/fairseq/fairseq/search.py#L281-L284

In this line we call RoBERTa on that tensor:

    ./story-gen-BART/fairseq/fairseq/search.py#L298

which returns a probability distribution over the vocabulary which we multiply by the coefficients here: 

    ./story-gen-BARTfairseq/fairseq/search.py#L307

and then add to the raw lprobs here:
      
     ./story-gen-BARTfairseq/fairseq/fairseq/search.py#L323-L324

ignore all the if learn lines in between as those are only activated if training coefficients



