Code for the EMNLP 2020 Paper [Content Planning for Neural Story Generation with Aristotelian Rescoring](https://www.aclweb.org/anthology/2020.emnlp-main.351.pdf).

If you use this code, please cite as:
```
@inproceedings{goldfarb-tarrant-etal-2020-content,
    title = "Content Planning for Neural Story Generation with Aristotelian Rescoring",
    author = "Goldfarb-Tarrant, Seraphina  and
      Chakrabarty, Tuhin  and
      Weischedel, Ralph  and
      Peng, Nanyun",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.351",
    doi = "10.18653/v1/2020.emnlp-main.351",
    pages = "4319--4338",
    abstract = "Long-form narrative text generated from large language models manages a fluent impersonation of human writing, but only at the local sentence level, and lacks structure or global cohesion. We posit that many of the problems of story generation can be addressed via high-quality content planning, and present a system that focuses on how to learn good plot structures to guide story generation. We utilize a plot-generation language model along with an ensemble of rescoring models that each implement an aspect of good story-writing as detailed in Aristotle{'}s Poetics. We find that stories written with our more principled plot-structure are both more relevant to a given prompt and higher quality than baselines that do not content plan, or that plan in an unprincipled way.",
}
```

Direct questions to [Seraphina](mailto:s.tarrant@ed.ac.uk)

# story-gen-BART

## Introduction

Story generation comprises of two stages: plot generation and story generation. Both stages involve a tuned BART model but have different inputs.

## Training
### Plot generation
```
Input: <story prompt>
Output: <story plot>
```

Story prompts are short writing commands that set up a scene, conflict, character, or all.
Writing prompts are tokenized. An example prompt is 
> You are an undead , resurrected unwillingly and controlled to serve as part of a necromancer 's army . Slowly , but steadily , you start to regain control of your body .

A plot is a plan for a story expressed as SRL tuples for each sentence. For example:

>   &lt;A0&gt; you &lt;V&gt; realized &lt;A1&gt; you ' ve blinked # &lt;A0&gt; you &lt;V&gt; blinked &lt;/s&gt; &lt;A1&gt; ent 0 &lt;V&gt; passes &lt;/s&gt; &lt;A2&gt; ent 0 &lt;V&gt; covers &lt;A1&gt; ent 1 # &lt;A1&gt; ent 1 &lt;V&gt; happens &lt;/s&gt; &lt;/s&gt; &lt;A1&gt; ent 2 &lt;V&gt; moves # &lt;A0&gt; you &lt;V&gt; track &lt;A1&gt; ent 2 &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; keep &lt;A1&gt; marching , onward and onward # &lt;A0&gt; You &lt;V&gt; marching &lt;/s&gt; &lt;A1&gt; Nothing &lt;V&gt; stops &lt;/s&gt; &lt;A1&gt; It &lt;V&gt; stops &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; your head &lt;A2&gt; to the left &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; continue &lt;A1&gt; counting # &lt;A0&gt; you &lt;V&gt; counting &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; see &lt;A1&gt; the others # &lt;A0&gt; you &lt;V&gt; make &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; watch &lt;A1&gt; the shadows change # &lt;A0&gt; you &lt;V&gt; recognize &lt;A1&gt; the crossroads # &lt;A0&gt; you &lt;V&gt; count &lt;A1&gt; more than the days &lt;/s&gt; &lt;V&gt; Growing &lt;A2&gt; more and more aware # &lt;A0&gt; ent 3 &lt;V&gt; keep &lt;A1&gt; going # &lt;A0&gt; ent 3 &lt;V&gt; going &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; stopped &lt;/s&gt; &lt;/s&gt; &lt;/s&gt; &lt;A1&gt; the others &lt;V&gt; turn # &lt;A1&gt; ent 3 &lt;V&gt; stopped # &lt;A1&gt; what &lt;V&gt; left &lt;A2&gt; of who you # &lt;A0&gt; you &lt;V&gt; need &lt;A1&gt; ent 4 to survive # &lt;A0&gt; you &lt;V&gt; survive # &lt;A0&gt; you &lt;V&gt; call &lt;A1&gt; ent 4 &lt;A2&gt; that &lt;/s&gt; &lt;A1&gt; what &lt;A0&gt; you &lt;V&gt; doing # &lt;A1&gt; the amount of roads &lt;A0&gt; you &lt;V&gt; pass # &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; ent 5 head # &lt;A1&gt; ent 5 head &lt;V&gt; left # &lt;V&gt; catch &lt;A1&gt; ent 6 bearings &lt;/s&gt; &lt;A0&gt; ent 3 &lt;V&gt; keep &lt;A1&gt; ent 3 your blinking # &lt;A0&gt; ent 3 &lt;V&gt; counting &lt;A1&gt; ent 3 blinking # &lt;A0&gt; ent 3 &lt;V&gt; blinking &lt;/s&gt; &lt;A1&gt; ent 7 &lt;V&gt; matter &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; fight &lt;A1&gt; the fog draped over you # &lt;A1&gt; the fog &lt;V&gt; draped &lt;A2&gt; over you # &lt;A1&gt; You &lt;V&gt; straining # &lt;A0&gt; you &lt;V&gt; keep &lt;A1&gt; moving on # &lt;A1&gt; you &lt;V&gt; moving &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; your head &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; clench &lt;A1&gt; your hands &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; feel &lt;A1&gt; stronger &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; control &lt;A1&gt; your feet  

The example comes from the `plot` directory which contains several more examples.

To generate plots, BART must learn from pairs of prompts and plots. The method in this paper adjusts the training process so tuning BART to generate plots involves extra loss terms. The supplementary loss terms are discriminators trained to encourage Aristotelian writing principles (see Aristotle's poetics on [Wikipedia](https://en.wikipedia.org/wiki/Poetics_(Aristotle)), the [original text](http://classics.mit.edu/Aristotle/poetics.1.1.html), or [explanations of the text](https://www.sparknotes.com/philosophy/poetics/summary/)).

### Fine-tuning prompt to plot

1. Use `encoder.json` and `dict.txt` already provided in the repo, since they contain additional delimeter tokens relevant for story generation.

2. Create a directory to store the plot data
     ```
      cd fairseq
      mkdir plot
     ```
 
    Since this is a seq2seq task you need source and target files. Put in four files in `plot` directory: `train.source`, `train.target`, `val.source`, `val.target`. Sample data is present in `fairseq/plot`.
 
 3. The next step is to tokenize the input using BPE tokens. You should download `vocab.bpe` from `fbaipublicfiles` with:
 
    ``` 
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    ``` 

    Now for BPE preprocess:
      ```
        sh create_bpe.sh <directory_name>
      ```

 4. Binarize dataset:

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

 5. Download Pretrained BART from https://github.com/pytorch/fairseq/tree/4b986c51ed89f9895df2f30e57909301b4a4f19b/examples/bart

 6. Fine tune BART:
    
    ```
    sh run.sh <data_dir> <save_dir>
    ```

    Update the field `BART_PATH` to point to where your pretained `model.pt` file is. You can customize `MAX_TOKENS` and `UPDATE_FREQ` based on GPU memory and number of GPUs.
    
    You should adjust the `--max-epoch 100` parameter. If you are running to verify that the code runs and finishes, set a low value (like 100). If you want to fully train the model, remove `--max-epoch`.
    
               To help in reproducibility , we have shared the data to finetune BART and finetuned PromptToPlot model
               In the STORYEMNLP folder "full" contains data required to train the model and checkpoint-full contains finetuned BART model
               https://drive.google.com/drive/folders/1cOouBxVsORnNdQJuZlH9fu3ACc7p9CwG?usp=sharing

### Story generation

#### Full outputs
For use in analysis and comparison to our system, the full set of outputs from the Aristotelian System and baselines used in the paper (Naive, and Prompt2Story) can be found here: https://drive.google.com/drive/folders/10VFDzJvH1ssByTch4UG8mh1DsTM0xpkI?usp=sharing 

Files ending in `.auto` are used for automatic evaluation (1000 stories). Files ending in `.human` were used for human evaluation (95 stories). Titles coindexed with the stories are in `title.auto` and `title.human.filtered`. Note that the stories and titles for human evaluation were filtered (refer to the paper for details) and have been detokenized. No modifications were made to the stories used for auto evaluation. We also include `title+plot` files so you can view the intermediate plot representation. 
Note that for the human title+plot files these have _not_ been filtered or detokenized, so they will be a superset of the `title.human.filtered` titles, and will not string match. 

#### Implementation

```
Input: <story plot>
Output: <story>
```

For example, given the story plot from the previous section::

>   &lt;A0&gt; you &lt;V&gt; realized &lt;A1&gt; you ' ve blinked # &lt;A0&gt; you &lt;V&gt; blinked &lt;/s&gt; &lt;A1&gt; ent 0 &lt;V&gt; passes &lt;/s&gt; &lt;A2&gt; ent 0 &lt;V&gt; covers &lt;A1&gt; ent 1 # &lt;A1&gt; ent 1 &lt;V&gt; happens &lt;/s&gt; &lt;/s&gt; &lt;A1&gt; ent 2 &lt;V&gt; moves # &lt;A0&gt; you &lt;V&gt; track &lt;A1&gt; ent 2 &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; keep &lt;A1&gt; marching , onward and onward # &lt;A0&gt; You &lt;V&gt; marching &lt;/s&gt; &lt;A1&gt; Nothing &lt;V&gt; stops &lt;/s&gt; &lt;A1&gt; It &lt;V&gt; stops &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; your head &lt;A2&gt; to the left &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; continue &lt;A1&gt; counting # &lt;A0&gt; you &lt;V&gt; counting &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; see &lt;A1&gt; the others # &lt;A0&gt; you &lt;V&gt; make &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; watch &lt;A1&gt; the shadows change # &lt;A0&gt; you &lt;V&gt; recognize &lt;A1&gt; the crossroads # &lt;A0&gt; you &lt;V&gt; count &lt;A1&gt; more than the days &lt;/s&gt; &lt;V&gt; Growing &lt;A2&gt; more and more aware # &lt;A0&gt; ent 3 &lt;V&gt; keep &lt;A1&gt; going # &lt;A0&gt; ent 3 &lt;V&gt; going &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; stopped &lt;/s&gt; &lt;/s&gt; &lt;/s&gt; &lt;A1&gt; the others &lt;V&gt; turn # &lt;A1&gt; ent 3 &lt;V&gt; stopped # &lt;A1&gt; what &lt;V&gt; left &lt;A2&gt; of who you # &lt;A0&gt; you &lt;V&gt; need &lt;A1&gt; ent 4 to survive # &lt;A0&gt; you &lt;V&gt; survive # &lt;A0&gt; you &lt;V&gt; call &lt;A1&gt; ent 4 &lt;A2&gt; that &lt;/s&gt; &lt;A1&gt; what &lt;A0&gt; you &lt;V&gt; doing # &lt;A1&gt; the amount of roads &lt;A0&gt; you &lt;V&gt; pass # &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; ent 5 head # &lt;A1&gt; ent 5 head &lt;V&gt; left # &lt;V&gt; catch &lt;A1&gt; ent 6 bearings &lt;/s&gt; &lt;A0&gt; ent 3 &lt;V&gt; keep &lt;A1&gt; ent 3 your blinking # &lt;A0&gt; ent 3 &lt;V&gt; counting &lt;A1&gt; ent 3 blinking # &lt;A0&gt; ent 3 &lt;V&gt; blinking &lt;/s&gt; &lt;A1&gt; ent 7 &lt;V&gt; matter &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; fight &lt;A1&gt; the fog draped over you # &lt;A1&gt; the fog &lt;V&gt; draped &lt;A2&gt; over you # &lt;A1&gt; You &lt;V&gt; straining # &lt;A0&gt; you &lt;V&gt; keep &lt;A1&gt; moving on # &lt;A1&gt; you &lt;V&gt; moving &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; turn &lt;A1&gt; your head &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; clench &lt;A1&gt; your hands &lt;/s&gt; &lt;A0&gt; You &lt;V&gt; feel &lt;A1&gt; stronger &lt;/s&gt; &lt;A0&gt; you &lt;V&gt; control &lt;A1&gt; your feet  

BART is tasked with generating the corresponding story. For the plot above, the *gold* story is:

> One day , you realized you ' ve blinked . &lt;/s&gt; ent 7 &lt;P&gt; &lt;P&gt; ent 0 A shadow passes in front of your eyes , frequently and [UNK] . &lt;/s&gt; ent 0 It covers ent 1 everything for a moment when ent 1 it happens . &lt;/s&gt; The others in front of you . &lt;/s&gt; The way ent 2 the sun moves in the sky- you can track ent 2 it now , counting the number of times your eyes open and close . &lt;/s&gt; You keep marching , onward and onward . &lt;/s&gt; Nothing stops . &lt;/s&gt; It never stops . &lt;/s&gt; &lt;P&gt; &lt;P&gt; Three weeks later , by your calculations , you can turn your head , ever so slightly , to the left . &lt;/s&gt; Stiff , but deliberate , you continue counting . &lt;/s&gt; You can see the others next to you , as far as you can make out . &lt;/s&gt; You watch the shadows change and you learn to recognize the crossroads as you past and you start to count more than the days . &lt;/s&gt; &lt;P&gt; &lt;P&gt; Growing more and more aware , ent 3 you keep going . &lt;/s&gt; If you stopped , you 'd be- &lt;P&gt; &lt;P&gt; Well , not dead . &lt;/s&gt; ent 3 You 're already dead . &lt;/s&gt; Or ent 3 you were . &lt;/s&gt; But the others would turn on you if ent 3 your stopped and part of what 's left of who you used to be knows that you need ent 4 the safety to survive , if you can call ent 4 it that . &lt;/s&gt; &lt;P&gt; &lt;P&gt; You know that what you 're doing is wrong , you can tell by the amount of roads you pass and how ent 6 the ones ahead of you are full and by the time you can turn ent 5 your head left enough to catch your bearings ent 6 they 're empty . &lt;/s&gt; ent 3 You keep counting ent 3 your blinking and ca n't help but wonder how many days it 's been since you were- ent 7 &lt;P&gt; &lt;P&gt; Nevermind . &lt;/s&gt; ent 7 That does n't matter . &lt;/s&gt; You fight the fog draped over you every day , straining as you keep moving on . &lt;/s&gt; Eventually you can turn your head to the right . &lt;/s&gt; You can clench your hands , ever so gently . &lt;/s&gt; You feel stronger . &lt;/s&gt; But you ca n't control your feet

Note that:
 * The text is tokenized.
 * Stories are split into sentences corresponding to the sentences in the SRL plot. In the training and validation data, the plots are generated from the stories, see *Section 2* in the paper.
 * Entities are corefereced throughout the story. This allows BART to output coherent stories.
 * New paragraphs are marked in the story text. The markers don't have any story value but are kept due to their presence in the original   *WritingPrompts* data set.

The example comes from the `story` directory which contains several more examples.

### Fine-tuning plot to story

1. Use `encoder.json` and `dict.txt` already provided in the repo, since they contain additional delimeter tokens relevant for story generation.

2. Create a directory to store the plot data
     ```
      cd fairseq
      mkdir story
     ```
 
    Since this is a seq2seq task you need source and target files. Put in four files in `plot` directory: `train.source`, `train.target`, `val.source`, `val.target`. Sample data is present in `fairseq/story`.
 
 3. The next step is to tokenize the input using BPE tokens. You should download `vocab.bpe` from `fbaipublicfiles` with:
 
    ``` 
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    ``` 

    Now for BPE preprocess:
      ```
        sh create_bpe.sh <directory_name>
      ```

 4. Binarize dataset:

      ```
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --trainpref "story/train.bpe" \
        --validpref "story/val.bpe" \
        --destdir "plot/" \
        --workers 60 \
        --srcdict dict.txt \
        --tgtdict dict.txt
      ```

 5. Fine tune BART. You can use the pretrained model downloaded during the prompt-to-plot training:
    
    ```
    sh run.sh <data_dir> <save_dir>
    ```

    Update the field `BART_PATH` to point to where your pretained `model.pt` file is. You can customize `MAX_TOKENS` and `UPDATE_FREQ` based on GPU memory and number of GPUs.
    
    You should adjust the `--max-epoch 100` parameter. If you are running to verify that the code runs and finishes, set a low value (like 100). If you want to fully train the model, remove `--max-epoch`.
    
    
               To help in reproducibility , we have shared the data to finetune BART and finetuned PlotToStory model
               In the STORYEMNLP folder "fullstory" contains data required to train the model and checkpoint-fullstory contains finetuned BART model
               https://drive.google.com/drive/folders/1cOouBxVsORnNdQJuZlH9fu3ACc7p9CwG?usp=sharing

### Train Aristotelian Rescorers (aka classifiers, aka discriminators)

To train a rescorer you will need to go through 3 steps:
1. Split the prompt + plot data into source and target. If you are using *WritingPrompts*, the data is already split into *train*, *valid*, and *test*.
2. Generate continuation data data. Use the script `preprocessing/make_cc_version_pnw_data.py`.
3. Compile TSV files for train and validation data. Use the script `preprocessing/create_classifier_dataset.py` to generate positive and negative examples and the TSV from the output of the previous step.
4. Preprocess task data and it needs binary .bin files to finetune roberta `./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>`. We use it as a proxy for RTE task (sentence pair with label 0 or 1) so recommend to use it.
5. Finetune the discriminator using RoBERTa-large. For fine-tuning on GLUE task use `run_disc.sh`.
    
    Finetune 4 discriminators:
    * 1.0 /nas/home/fairseq/relevance-roberta/  checkpoint_best.pt  roberta.large/
    * 1.0 /nas/home/fairseq/eventinter-roberta  checkpoint_best.pt  roberta.large/
    * 1.0 /nas/home/fairseq/eventintraV-roberta checkpoint_best.pt  roberta.large/
    * 1.0 /nas/home/fairseq/entity-roberta  checkpoint_best.pt  roberta.large/


### Mixture Weight training for the Aristotelian rescorers

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


## Inference

#### Prompt-to-plot:

```
python inference_plot.py
```

Calling the script with `--help` shows all the required arguments.

The script needs `dict.source.txt` and `dict.target.txt` to be copied from the `plot` directory to the checkpoint directory.

If you want to use the aristotelian rescorers, look at the arguments `--apply_disc` and `--scorers`.

You might need to install `requests` from `pip`.

#### Plot-to-story:

```
python inference_story.py
```

Calling the script with `--help` shows all the required arguments.

The script needs `dict.source.txt` and `dict.target.txt` to be copied from the `story` directory to the checkpoint directory.

You might need to install `requests` from `pip`.

