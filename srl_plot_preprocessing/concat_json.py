import json
import os
import io

def concat(inpath,outdir):

    files = os.listdir(inpath)

    files_txt = [os.path.join(inpath, i) for i in files if i.endswith('.json')]
    all = []
    for file in files_txt:
        with open(file, "r") as fin:
            data = json.load(fin)
            for item in data:
                all.append(item)

    with io.open(outdir, "w", encoding='utf8') as fout:
        json.dump(all, fout, ensure_ascii=False)


if __name__ == "__main__":
    inpath = "data/writingPrompts/srl_output/story"
    outdir = "data/writingPrompts/srl_output/WP.story.train.json"
    concat(inpath,outdir)