#!/usr/bin/env bash

# run from root dir 
types="test train valid"
parent_dir="data/WritingPrompts/fairseq_baseline/"

# generate storylines
for type in ${types}; do
    python python_src/rake_story_keywords.py ${parent_dir}WP.story.${type} 5 1 1 ${parent_dir}WP.storyline_metrics.${type} ${parent_dir}WP.storyline.${type}  "punkt"
done

# concat storylines and titles and stories
for type in ${types}; do
    python preprocessing/generate_WP_training_files.py ${parent_dir}WP.title.${type} ${parent_dir}WP.storyline.${type} ${parent_dir}WP.story.${type} ${type} ${parent_dir}
done
