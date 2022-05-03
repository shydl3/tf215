stopWordsFile = open("descStopWords.txt", "r")
descFile = open("DeepCom_JAVA/test_IR_code_desc.txt", "r", encoding="utf-8")
finalFile = open("DeepCom_JAVA/test_IR_code_desc_sw.txt", "w", encoding="utf-8")
stopwords = stopWordsFile.readlines()
wordList = list([])
for word in stopwords:
    word = word.strip()
    wordList.append(word)

for i, line in enumerate(descFile):
    string = ""
    words = line.split()
    for w in words:
        if w not in wordList:
            string = string + w + " "
    finalFile.write(string.strip() + "\n")
    print(i)
stopWordsFile.close()
descFile.close()
finalFile.close()
