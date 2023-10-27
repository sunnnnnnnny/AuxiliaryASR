import os
import random
from glob import glob
from sklearn.model_selection import train_test_split

random.seed(616)

# parameter dir
data_dir = "../Data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
wav_dir = "/data8/master_dataset_model_zhangyuqiang/aishell3_24k_without_normal"

wav_path_list_tmp = glob(wav_dir + "/*/*.wav")
print(len(wav_path_list_tmp))

filename2piny = {}
with open("/data8/master_dataset_model_zhangyuqiang/aishell3_44k/train/content.txt", "r") as log:
    lines = log.readlines()
    for line in lines:
        filename, sentence = line.strip().split(" ", maxsplit=1)
        piny = " ".join(sentence.split(" ")[1::2])
        filename2piny[filename] = piny
with open("/data8/master_dataset_model_zhangyuqiang/aishell3_44k/test/content.txt", "r") as log:
    lines = log.readlines()
    for line in lines:
        filename, sentence = line.strip().split(" ", maxsplit=1)
        piny = " ".join(sentence.split(" ")[1::2])
        filename2piny[filename] = piny



wav_path_list = []
wav_label_list = []

for wav_path in wav_path_list_tmp:
    txt_path = wav_path.replace(".wav", ".txt")
    if os.path.exists(txt_path):
        f = open(txt_path, "r")
        lines = f.readlines()
        if len(lines) > 0:
            wav_path_list.append(wav_path)
            label = lines[0].strip()
            wav_label_list.append(label)

assert len(wav_label_list) == len(wav_path_list)

train_path_list, eval_path_list, train_label_list, eval_label_list = train_test_split(wav_path_list, wav_label_list,
                                                                                      test_size=0.2, random_state=616)
assert len(train_path_list) == len(train_label_list)
assert len(eval_path_list) == len(eval_label_list)

with open(os.path.join(data_dir, "train_list_aishell3.txt"), "w") as train_log:
    for idx in range(len(train_path_list)):
        path = train_path_list[idx]
        filename = path.split("/")[-1]
        if filename not in filename2piny:
            continue
        label = filename2piny[filename]
        line = path + "|" + label + "|something\n"
        train_log.write(line)

with open(os.path.join(data_dir, "eval_list_aishell3.txt"), "w") as eval_log:
    for idx in range(len(eval_path_list)):
        path = eval_path_list[idx]
        filename = path.split("/")[-1]
        if filename not in filename2piny:
            continue
        label = eval_label_list[idx]
        line = path + "|" + label + "|something\n"
        eval_log.write(line)