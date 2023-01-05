from sklearn.feature_extraction.text import TfidfVectorizer
import csv

n = "1" #ngramのn
file_name = f'./solution_{n}gram.csv' #読み込むファイル
save_file = f'./idf_{n}gram.csv' #書き込むファイル

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

#idfを計算する
def idf_cal(ngram_list):
    lines_list = []
    label_list = []
    for l in ngram_list:
        label_list.append([l[0]])
        lines_list.append(l[1])
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
    vectorizer.fit(lines_list)
    word = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    return word, label_list, lines_list, idf

#1回しか出現していない単語を削除
def selection_word(word, lines_list):
    word1_dict = []
    word2_dict = []
    result = word.tolist()
    for line in lines_list:
        line = line.split()
        for w in line:
            if w not in word1_dict and w not in word2_dict:
                word1_dict.append(w)
            elif w in word1_dict and w not in word2_dict:
                word1_dict.remove(w)
                word2_dict.append(w)
            else:
                pass
    for w in word1_dict:
        try:
            result.remove(w)
        except:
            pass
    return result
    
#idfを並び替える
def idf_sort(word, label_list, lines_list, idf):
    word = word.tolist()
    word_list = []
    posi_list = []
    result_list = label_list
    #ソースコードの単語リスト作成と単語の位置リストの仮作成
    for l in lines_list:
        tmp_word = l.split()
        word_list.append(tmp_word)
        posi_list.append([0 for i in range(len(tmp_word))])
    #単語の位置リスト作成
    for i in range(len(posi_list)):
        c = 0
        for w in word_list[i]:
            try:
                posi_list[i][c] = word.index(w.lower())
            except:
                posi_list[i][c] = -1
            c += 1
    #単語の位置リストからreturnするリストにidf情報を入れる並び替え
    empty_min = 100000
    for i in range(len(result_list)):
        empty = 0
        for j in range(len(word)):
            try:
                posi = posi_list[i].index(j)
                result_list[i].append(idf[posi])
            except:
                empty += 1
        if empty_min > empty:
            empty_min = empty
        for j in range(empty):
            result_list[i].append(0)
    return [l[:-empty_min] for l in result_list]

if __name__ == "__main__":
    ngram_list = read_csv(file_name)
    word, label_list, lines_list, idf = idf_cal(ngram_list)
    #word = selection_word(word, lines_list)
    print(len(word))
    output_list = idf_sort(word, label_list, lines_list, idf)
    write_csv(save_file, output_list)