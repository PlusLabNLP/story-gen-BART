# story-gen-BART

Use the encoder.json and dict.txt already provided as a part of the repo, since it contains additional delimeter tokens relevant for story generations

 ```
  cd fairseq
  mkdir temp
 ```
 
 Since this is a seq2seq task you need source and target files
  - Put in temp directory 4 files train.source, train.target, val.source, val.target
 
 Now for BPE preprocess:
  ```
    sh create_bpe.sh
  ```

Binarize dataset:

      ```
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --trainpref "temp/train.bpe" \
        --validpref "temp/val.bpe" \
        --destdir "temp/" \
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



For Inference:

  ```
    python inference.py
  ```
 
## Train Discriminators aka Classifier aka whatever we start calling them

To train a discriminator you will need to go through 3 steps:
1) generate positive and negative example data
2) compile tsv files for train and validation data
3) train the discriminator

Step one:
Run the below command. `disc_data_dir` should have three files in it for disc_train.txt, valid.txt, test.txt

`python preprocessing/make_cc_version_pnw_data.py ${disc_data_dir} --out_dir ${disc_data_dir} \
--sent_sym "</s>" --len_context 1 --keep_split_context`
This generates different types of continuation data from which to assemble training data for the existing discriminators.

Step two:
This is an example for a relevance discriminator. Use different `--comp` and `--event` values for different kinds of data. `rel_data_dir` is a new empty directory for relevance data - there should be a different dir per discriminator.

`python preprocessing/create_classifier_dataset.py ${disc_data_dir} ${rel_data_dir} --comp random`

Step three:
The output directory from the previous step will have tsvs in it that will be read in to train a new discriminator.

`python train_classifier.py disc_train/relevance --save_to checkpoint/rel_test.pt --decider_type cnncontext --lr 0.001 --num_epochs 100 --p --cuda --train_prefixes --batch_size 2 --max_seq_len 200`

Note:
*Currently the code only supports cnncontext, though there are two other types of discriminator (pool ending and reprnn) that we could support.
*Processing all the data to use in training with the bart encoder takes a _long_ time. So it will automatically save a processed corpus with the timestamp it was created. If you already have one, load it in via --load_corpus NAME
*Note the tiny batch size. At larger batch sizes, BART tends to run out of memory, but this is related to the sequence length max as well, so you can vary that
*you can experiment with different types of loss - check out the available options in the script.

