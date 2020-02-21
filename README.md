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
 


