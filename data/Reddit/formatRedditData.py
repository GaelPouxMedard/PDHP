import os
import numpy as np

means = np.array([0.5, 1, 8, 12, 24, 48, 72, 96, 120, 144, 168])
sigs = np.array([1, 1, 8, 12, 12, 24, 24, 24, 24, 24, 24])
lamb0 = 0.1

listFiles = os.listdir(".")
listFiles = [file for file in listFiles if ".txt" in file and "news_titles_3" in file]
#listFiles = [file for file in listFiles if "-04" in file]
#listFiles = listFiles[:6]
print(listFiles)


id_to_index = {}
id_to_metadata = {}
word_to_num = {}
index = 0
num_words = 0

cnt_wds = {}
for file in listFiles:
    print(file)
    with open(file, "r", encoding="utf-8") as fin:
        for line in fin:
            l = {}
            try:
                id, timestamp, l["subreddit"], title, l["selftext"], l["score"], l["num_crossposts"], l["num_comments"], l[
                    "permalink"] = line.replace("\n", "").split("\t")
            except:
                print(len(line.replace("\n", "").split("\t")), line)

            for wd in title.split(" "):
                if wd not in cnt_wds: cnt_wds[wd] = 0
                cnt_wds[wd] += 1

cnt = np.array(list(cnt_wds.values()))
thres = 5
print(len(cnt[cnt>thres]))

with open("Reddit_3_events.txt", "w+", encoding="utf-8") as fout:
    for file in listFiles:
        print(file)
        with open(file, "r", encoding="utf-8") as fin:
            for line in fin:
                l = {}
                try:
                    id, timestamp, l["subreddit"], title, l["selftext"], l["score"], l["num_crossposts"], l["num_comments"], l["permalink"] = line.replace("\n", "").split("\t")
                except:
                    print(len(line.replace("\n", "").split("\t")), line)
                timestamp = str(float(timestamp)/3600)  # heures
                if id not in id_to_index:
                    id_to_index[id] =index
                    index += index

                if id not in id_to_metadata:
                    id_to_metadata[id] = [l["subreddit"], l["selftext"], l["score"], l["num_crossposts"], l["num_comments"], l["permalink"]]

                txt = "-1\t-1\t"+str(timestamp)+"\t"  # -1 -1 bc of my stupid ass data formatting, unimportant
                hasWds = False
                for wd in title.split(" "):
                    if cnt_wds[wd]<=thres:
                        continue
                    if wd not in word_to_num:
                        word_to_num[wd] = num_words
                        num_words += 1
                    if wd != "\n" and wd != " " and wd != "":
                        txt += str(word_to_num[wd])+","
                        hasWds = True

                if hasWds:
                    txt = txt[:-1] + "\n"
                    fout.write(txt)


with open("Reddit_3_metadata.txt", "w+", encoding="utf-8") as fout:
    for id in id_to_metadata:
        txt = id+"\t"+"\t".join(id_to_metadata[id])+"\n"
        fout.write(txt)
with open("Reddit_3_words.txt", "w+", encoding="utf-8") as fout:
    for wd in word_to_num:
        txt = str(word_to_num[wd])+"\t"+wd+"\n"
        fout.write(txt)

with open("Reddit_3_lamb0.txt", "w+") as f:
    f.write(str(lamb0))

np.savetxt("Reddit_3_means.txt", means)
np.savetxt("Reddit_3_sigs.txt", sigs)




