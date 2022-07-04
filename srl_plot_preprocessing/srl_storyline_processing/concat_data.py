
import glob, os
import io

def load(input):
    with open (input, 'r') as fin:
        data = fin.read()

    return data



if __name__ == "__main__":
    dir = "data/writingPrompts/srl_resume/train/story/"
    # print(glob.glob("data/writingPrompts/srl_resume/train/story/*",recursive=True))
    file_list = glob.glob("data/writingPrompts/srl_resume/train/story/*",recursive=True)
    sort_file = sorted(file_list, key=lambda i: (os.path.split(i)[1].split('.')[-2]))
    print(sort_file)
    # print(file_list)
    # files = os.listdir(dir,recursive=True)
    # files_txt = [i for i in files if i.endswith('.txt')]
    # files.sort()
    # print(files)
    list = []

    for file in sort_file:
        print(file)
        file_name = os.path.split(file)[1]
        # print('Processing {}'.format(file_name))
        data = load(os.path.join(dir, file_name))
        # print(data)
        list.append(data)

    all = '\n'.join(list)
    with open ('data/writingPrompts/srl_resume/train/plot.story.txt','w') as fout:
        fout.write(all)
