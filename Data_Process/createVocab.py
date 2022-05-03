from collections import Counter

vocabWord = open("DeepCom_JAVA/corpus_tokens.txt", "r", encoding="utf-8")
processStaFile = open("DeepCom_JAVA/vocab.tokens.txt", "w", encoding="utf-8")
staTreeList = []
while 1:
    word = vocabWord.readline().splitlines()
    if not word:
        break
    staTreeList.append(word[0])

staTreeDic = Counter(staTreeList)

for k, v in staTreeDic.items():
    if v >= 11:
        processStaFile.write(k + '\n')