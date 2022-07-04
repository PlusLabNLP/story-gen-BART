import os


def load(dir, file):
    with open(os.path.join(dir, file), "r") as fin:
        data = fin.read()
        length = len(data.split("\n"))
    return data,length



dir = "/Users/yangjinrui/Documents/summer/storyGeneration/Plan-and-write/data/writingPrompts/srl_output/ready_train/ready_train_model"
files = ["WP.titlesepkey.train", "WP.titlesepkey.valid", "WP.titlesepkey.test"]
all = ""
for file in files:
    data,length = load(dir, file)
    print("{0}:{1}".format(file, length))
    all += data + "\n"
print("all:{}".format(len(all.split("\n"))))
with open(os.path.join(dir, "WP.titlesepkey.all"), "w") as fout:
    fout.write(all)