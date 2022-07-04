# Processing Storyline

Note: this is *Old* code using previous versions of SRL models, written by an intern at ISI. There has been major amounts of improvement in SRL since, so I recommend you switch to a new bert based srl, which allennlp now provides. You'll have to update the version, and hopefully their coref models are now compatible with their bert srl (at the time of publication they were not but that was ages ago).

So ideally this code will be only for reference on how to do it, but use new models.

These scripts are used for processing storyline based on semantic role labeling and coreference resolution.

## Files

The files' purposes in this folder.


 ## 1. /srl_to_storyline.py
This .py contains all main code to run srl and coref to generate storyline, the input is story, and output is a dictionary of storyline and if you use --save_core_srl, it will save all parsed coreference and srl information for future use, the default is on, you can delete it if you don't wanna to save them.
## 2. /prepare_SRL_storyline_format.py
This .py is used for change storyline format and delete "{}", extract all use info from the dictionary to a string. and use "#" to separate.


## 3. /generate_WP_training_files_srl.sh
This .sh is bash file to run all script to make data ready to train. you can change parameters here.



# Requirements
In oreder to do storyline processing basing on SRL+Coref, you need meet these requirements.

## 1. Packages
Use the [pip](https://pip.pypa.io/en/stable/) to install these packages you may not install.

```bash
pip install allennlp
pip install tqdm
pip install time
pip install json
```
## 2. Data
 Download the story and title which have been general preprocessed from [google drive](https://drive.google.com/drive/folders/1y4OBcl4BUq1fL5A261QFAy6LS7rbn-jf).
Strongly recommend you download it, since I wash the baic_processed data again, and filter some [EU][PI] tag, making the separating sentence better. Also, suggest you put these six files into:
```
/data/writingPrompts/ready_srl/
```
otherwise, you need to modify the parent_dir to your own path in line6 in
```
/generate_WP_training_files_srl.sh
```

## 3. Pretrained model
 Download the pre-trained model for SRL and coreference resolution from [google drive](https://drive.google.com/drive/folders/1BSQrcwzerAUt6RU-qQSPHMEp-5OJ8TEs).
Also, suggest you put these two folders into:
```
/preprocessing/srl_storyline_processing/pretrained/
```
otherwise, you need to modify the coref_model and srl_model to your own path in line7, 8 in
```
/generate_WP_training_files_srl.sh
```


## Usage
With all done below, you just need to run this command:

```bash
bash preprocessing//srl_storyline_processing/generate_WP_training_files_srl.sh
```
WTR the batch size and Cuda device, you can change the parameters at the end of line18 in

```
/generate_WP_training_files_srl.sh
```
PS: according to the Document of Allennlp, the argument --cuda refer to the cuda device id. The type of this parameter is int, different from pytorch which is a boolean, so if we wanna use GPU, we need set --cuda 0, otherwise, CPU is --cuda -1.
