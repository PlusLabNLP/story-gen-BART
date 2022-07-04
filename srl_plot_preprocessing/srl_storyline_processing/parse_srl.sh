#!/usr/bin/env bash


# run from root dir
cd ~/PycharmProjects/Plan-and-write/
types="random"

#types="sample test valid train"
parent_dir="data/writingPrompts/ready_srl/"
coref_model="preprocessing/srl_storyline_processing/pretrained/coref-model-2018.02.05"
srl_model="preprocessing/srl_storyline_processing/pretrained/srl-model-2018.05.25"


#generate storylnes by SRL
for type in ${types}; do
    echo ${type}
    python preprocessing/srl_storyline_processing/srl_to_storyline.py --input_file ${parent_dir}WP.titlesepstory.${type} --output_file ${parent_dir}WP.storyline_dic.${type}.json --save_coref_srl ${parent_dir}WP.prediction.${type}.json --label_story ${parent_dir}WP.valid_story.${type}.json --title ${parent_dir}WP.valid_title.${type}.json --coref_model ${coref_model} --srl_model ${srl_model} --batch 256 --cuda -1
done

