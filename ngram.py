import csv

n = "3" #ngramのn
file_name = './solution_python_processing.csv' #読み込むファイル
save_file = f'./solution_{n}gram.csv' #書き込むファイル

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

#unigramを適用
def unigram(code_list):
    return code_list

#bigramを適用
def bigram(code_list):
    word_list = []
    output_list = []
    for l in code_list:
        word_list.append(l[1].split())
    for i in range(len(word_list)):
        bigram_list = []
        for j in range(len(word_list[i])-1):
            bigram_list.append(word_list[i][j] + word_list[i][j+1])
        output_list.append([code_list[i][0], " ".join(bigram_list)])
    return output_list

#trigramを適用
def trigram(code_list):
    word_list = []
    output_list = []
    for l in code_list:
        word_list.append(l[1].split())
    for i in range(len(word_list)):
        trigram_list = []
        for j in range(len(word_list[i])-2):
            trigram_list.append(word_list[i][j] + word_list[i][j+1] + word_list[i][j+2])
        output_list.append([code_list[i][0], " ".join(trigram_list)])
    return output_list

#関数実行諸々
if __name__ == "__main__":
    code_list = read_csv(file_name)
    if n == "1":
        ngramlist = unigram(code_list)
    elif n == "2":
        ngramlist = bigram(code_list)
    elif n == "3":
        ngramlist = trigram(code_list)
    write_csv(save_file, ngramlist)