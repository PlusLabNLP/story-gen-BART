#!/usr/bin/env bash

cd ~/PycharmProjects/Plan-and-write/
# run from root dir
types="test valid train"
parent_dir="data/writingPrompts/original/test/"
coref_model="preprocessing/pretrained/coref-model-2018.02.05"
srl_model="preprocessing/pretrained/bert-base-srl-2019.06.17" #"pretrained/srl-model-2018.05.25"

## generate storylines
#for type in ${types}; do
#    python python_src/rake_story_keywords.py ${parent_dir}WP.story.${type} 5 1 1 ${parent_dir}WP.storyline_metrics.${type} ${parent_dir}WP.storyline.${type}  "punkt"
#done


#generate storylnes by SRL
echo "Extracting SRL Plots..."
for type in ${types}; do
    python preprocessing/srl_storyline_processing/srl_to_storyline.py --input_file ${parent_dir}${type}.wp_target --output_file ${parent_dir}WP.storyline_dic.${type}.json --save_coref_srl ${parent_dir}WP.srl_coref.${type}.json --label_story ${parent_dir}WP.label_story.${type} --coref_model ${coref_model} --srl_model ${srl_model} --batch 8 --cuda -1
done

echo "Assembling final file format..."
#change storyline format for dictionary to just string
for type in ${types}; do
    python preprocessing/srl_storyline_processing/prepare_SRL_storyline_format.py --input_file ${parent_dir}WP.storyline_dic.${type}.json --output_file ${parent_dir}WP.storyline.${type}
done

#for type in ${types}; do
#    python preprocessing/srl_storyline_processing/prepare_SRL_storyline_format.py --input_file ${parent_dir}WP.valid_story.${type}.json --output_file ${parent_dir}WP.story.${type}
#done

#for type in ${types}; do
#    python preprocessing/srl_storyline_processing/prepare_SRL_storyline_format.py --input_file ${parent_dir}WP.valid_title.${type}.json --output_file ${parent_dir}WP.title.${type}
#done

# concat storylines and titles and stories
#for type in ${types}; do
#    echo ${type}
#    python preprocessing/srl_storyline_processing/concact.py  --title_file ${parent_dir}WP.title.${type} --kw_file ${parent_dir}WP.storyline.${type} --story_file ${parent_dir}WP.story.${type} --data_type ${type} --target_dir ${parent_dir} --title_story
#done
