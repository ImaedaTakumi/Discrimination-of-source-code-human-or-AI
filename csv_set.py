import csv
import pandas as pd

def sub_text(text,sub_dict):
    sub_text = str(text)
    for key,value in sub_dict.items():
        sub_text = sub_text.replace(key,f" {value} ")
    return sub_text

# replace word list
subdict = {"\r":" ",
"\n":" ",
"\r\n":" ",
"(":"leftround",
")":"rightround",
"[":"leftsquare",
"]":"rightsquare",
">":"rightangle",
"<":"leftangle",
"{":"leftcurly",
"}":"rightcurly",
":":"colon",
";":"semicolon",
'"':"d-quotation",
"'":"s-quotation",
".":"period",
",":"comma",
"|":"pipe",
"&":"ampersand",
"%":"percent",
"=":"equal",
"+":"plus",
"-":"minus",
"*":"asterisk",
"^":"hat",
"/":"slash",
"#":"hash",
"!":"exclamation",
"?":"question",
"@":"at",
"$":"doller"
}

processed_list = []

# source code written by human
filename = "solution_python.csv"
df = pd.read_csv(filename)
label = 0
for raw in df["Solutions"]:
    data = sub_text(raw,subdict)
    data = data.split()
    data = " ".join(data)
    processed_list.append([label,data])
    
# source code written by AI
filename = "solution_python_ai.csv"
df = pd.read_csv(filename)
label = 1
for raw in df["Solutions"]:
    data = sub_text(raw,subdict)
    data = data.split()
    data = " ".join(data)
    processed_list.append([label,data])

savefile = "solution_python_processing.csv"
with open(savefile,mode="a",encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(processed_list)