if __name__ == '__main__':
    sourceFile = open("DeepCom_JAVA/train_tokens.txt", "r", encoding="utf-8")
    corpusFile = open("DeepCom_JAVA/corpus_tokens.txt", "w", encoding="utf-8")
    for num, line in enumerate(sourceFile):
        print(num)
        words = line.split()
        for word in words:
            corpusFile.write(word + '\n')
    sourceFile.close()
    corpusFile.close()
