import string

punctuations = list(string.punctuation)
import spacy
#spacy_nlp = spacy.load('en_core_web_trf')
spacy_nlp = spacy.load('en_core_web_sm')
toreject = ["DET", "AUX", "PUNCT", "ADP", "CCONJ", "PRON", "SCONJ", "PART"]

import zstandard as zstd
import json
import os

def preprocess_spacy(txt):
    txt = spacy_nlp(txt)
    txt = " ".join([w.lemma_.lower() for w in txt if w.pos_ not in toreject]).replace("\n", " ")
    txt.replace("  ", " ")
    return txt

def treatZST(folder, listFiles, subsToConsider):
    # created_utc, id, num_comments, num_crossposts,
    # permalink, score, selftext, subreddit, title

    for file in listFiles:
        with open(folder+file, 'rb') as fh:
            s = os.path.getsize(folder+file)
            r = 50643213381/4686812440
            sd = s*r  # Approx
            print()
            print(folder+file)

            with open(f"news_titles_4_{file.replace('.zst', '')}.txt", "w+", encoding="utf-8") as fout:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    previous_line=""
                    iterChunk = 0
                    while True:
                        chunk = reader.read(2**27)
                        iterChunk+=1

                        if not chunk:
                            break

                        print(reader.tell()*100/sd, "%")

                        string_data = chunk.decode('utf-8')

                        lines = string_data.split("\n")

                        for iter, line in enumerate(lines[:-1]):
                            if iter == 0:
                                line = previous_line + line

                            l = json.loads(line)

                            if l["subreddit"] not in subsToConsider:
                                continue

                            l["title"] = preprocess_spacy(l["title"])
                            l["selftext"] = preprocess_spacy(l["selftext"])
                            tup = [l["id"], l["created_utc"], l["subreddit"], l["title"], l["selftext"], l["score"], l["num_crossposts"], l["num_comments"], l["permalink"]]
                            tup = [str(t) for t in tup]
                            fout.write("\t".join(tup)+"\n")
                            print(l["created_utc"])
                            pause()


                        previous_line = lines[-1]


folder = "Raw/"
listFiles=os.listdir(folder)
listFiles = [file for file in listFiles if "zst" in file]
'''
subsToConsider = ["anythinggoesnews", "inthenews", "nottheonion", "offbeat", "onthescene", "qualitynews", "news",
                  "thenews", "upliftingnews", "USNews", "Full_news", "FullNEWS", "neutralnews", "anime_titties",
                  "worldnews", "truenews", "open_news"]
'''
subsToConsider = ['inthenews', 'neutralnews', 'news', 'nottheonion', 'offbeat',
                  'open_news', 'qualitynews', 'truenews', 'worldnews']
subsToConsider = ['todayilearned']
subsToConsider = ['askscience']
subsToConsider = ['france', "canada"]
subsToConsider = [sub.lower() for sub in subsToConsider]

treatZST(folder, listFiles, subsToConsider)