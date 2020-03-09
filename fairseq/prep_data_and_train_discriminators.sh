#!/usr/bin/env bash

#### Please configure the directories based on your folder structure. You may not need all of them depending on 
# which discriminators you are training. 

cd /Users/seraphinagoldfarb-tarrant/PycharmProjects/Plan-and-write/

disc_data_dir="data/WritingPrompts/srl_storyline_data/unnested_version_data/"  #This is where all of the pre-split data should be, entitled train.txt, disc_train.txt, valid.txt, test.txt
#rep_data_dir="${disc_data_dir}repetition/"
rel_data_dir="${disc_data_dir}relevance/"
event_data_dir="${disc_data_dir}event_order/"

model_dir="models/"
#model="YOUR_MODEL_NAME"
#temp=0.8
script_dir="pytorch_src/"


######### DATA PREP ##############

# create data training files for classifiers - splits into context and continuations
python preprocessing/make_cc_version_pnw_data.py ${disc_data_dir} --out_dir ${disc_data_dir} \
--sent_sym "</s>" --len_context 1 --keep_split_context --filenames disc_train.txt.plot disc_train.txt.title+plot

# make generated continuations for use with pretrained models (this can take quite awhile). 
#It is only necessary for the Repetition Discriminator
#python example_scripts/aaai_scripts/gen_storylines.sh ${temp}

# Relevance Classifier needs preprocessed data
# Relevance
mkdir -p ${rel_data_dir}
python preprocessing/create_classifier_dataset.py ${disc_data_dir} ${rel_data_dir} --comp random --filenames disc_train.txt.plot disc_train.txt.title+plot

# Event Order
mkdir -p ${event_data_dir}
for type in intraV intra inter
    do mkdir -p ${event_data_dir}/${type}
    python preprocessing/create_classifier_dataset.py ${disc_data_dir} ${event_data_dir}/${type} --comp event --filenames disc_train.txt.plot disc_train.txt.title+plot --event_shuffle ${type}  # or intra/inter
done

############# TRAIN STEP ###########
# Repetition
#echo "Training Repetition discriminator"
#python ${script_dir}train_classifier.py ${rep_data_dir} --save_to ${model_dir}rep.pt \
#--dic ${model_dir}${model}.pkl --decider_type reprnn --train_prefixes --fix_embeddings --adam --lr 0.01  --num_epochs 50 --p --ranking_loss

# Lexical Style
#echo "Training Lexical Style discriminator"
#python ${script_dir}train_classifier.py ${rep_data_dir} --save_to ${model_dir}lex.pt --dic ${model_dir}${model}.pkl --decider_type poolending --lr 0.01  --num_epochs 50 --fix_embeddings --p --ranking_loss --adam

# Relevance
#echo "Training Relevance discriminator"
#python ${script_dir}train_classifier.py ${rel_data_dir} --save_to ${model_dir}rel.pt \
#--dic ${model_dir}${model}.pkl --decider_type cnncontext --train_prefixes --lr 0.001  --num_epochs 100 --fix_embeddings --p

# Event Ordering
#echo "Training Event Ordering discriminator"
#python ${script_dir}train_classifier.py ${event_data_dir} --save_to ${model_dir}event.pt \
#--dic ${model_dir}${model}.pkl --decider_type cnncontext --train_prefixes --lr 0.001  --num_epochs 100 --fix_embeddings --p

