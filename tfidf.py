from sklearn.feature_extraction.text import TfidfVectorizer
import csv

n = "3" #ngramのn
file_name = f'./solution_{n}gram.csv' #読み込むファイル
save_file = f'./tfidf_{n}gram.csv' #書き込むファイル

#csvファイルを読み込む
def read_csv(file_name):
    output_list = []
    csv_file = open(file_name, "r", encoding="utf-8")
    f = csv.reader(csv_file, delimiter=",")
    for row in f:
        output_list.append(row)
    csv_file.close()
    return output_list

#csvファイルに書き込む
def write_csv(save_file, output_list):
    with open(save_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(output_list)

#tfidfを求める
def tfidf_cal(ngram_list):
    lines_list = []
    label_list = []
    for l in ngram_list:
        label_list.append([l[0]])
        lines_list.append(l[1])
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
    data = vectorizer.fit_transform(lines_list)
    word = vectorizer.get_feature_names_out()
    tfidf = data.toarray().tolist()
    return word, label_list, lines_list, tfidf

#tfidfを時系列順に
def sort_tfidf(word, label_list, lines_list, tfidf):
    posi_list = [[0 for i in range(len(tfidf[0]))] for i in range(len(tfidf))]
    word_list = []
    for l in lines_list:
        word_list.append(l.split())
    for i in range(len(tfidf)):
        c = 0
        for w in word:
            try:
                posi_list[i][c] = word_list[i].index(w.lower())
            except:
                posi_list[i][c] = -1
            c += 1
    result = label_list
    empty_min = 100000
    for i in range(len(result)):
        empty = 0
        for j in range(len(tfidf[0])):
            try:
                posi = posi_list[i].index(j)
                result[i].append(tfidf[i][posi])
            except:
                empty += 1
        if empty_min > empty:
            empty_min = empty
            print(empty_min)
        for j in range(empty):
            result[i].append(0)
    return [l[:-empty_min] for l in result]

#特徴量飛びを調べて出力するコード
def check(word, lines_list):
    loss_list = []
    for l in lines_list:
        tmp = l.split()
        for w in tmp:
            if w.lower() in word:
                pass
            else:
                if w.lower() not in loss_list:
                    loss_list.append(w.lower())
    
    with open("loss.txt", "w", encoding="utf-8") as f:
        output = "\n".join(loss_list)
        f.write(output)
    with open("remain.txt", "w", encoding="utf-8") as f:
        output = "\n".join(word)
        f.write(output)
#以下ファイルネーム入れて実行するだけ関数一覧
#初期案のtfidf計算＆出力コード
def default(file_name):
    ngram_list = read_csv(file_name)
    word, label_list, lines_list, tfidf = tfidf_cal(ngram_list)
    check(word, lines_list)
    tfidf_list = sort_tfidf(word, label_list, lines_list, tfidf)
    write_csv(save_file, tfidf_list)

#tfidf時系列化しない
def non_time_series(file_name):
    ngram_list = read_csv(file_name)
    word, label_list, lines_list, tfidf = tfidf_cal(ngram_list)
    check(word, lines_list)
    tfidf_list = []
    for i in range(len(label_list)):
        tmp_list = [int(label_list[i][0])]
        tmp2_list = [j for j in tfidf[i]]
        for j in range(len(tmp2_list)):
            tmp_list.append(tmp2_list[j])
        tfidf_list.append(tmp_list)
    write_csv(save_file, tfidf_list)

#関数実行諸々
if __name__ == "__main__":
    default(file_name)