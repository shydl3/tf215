import codecs
import pickle

sbt_list = []
f = codecs.open('DeepCom_JAVA/vocab.tokens.txt', encoding='utf8', errors='replace').readlines()

for line in f:
    line = line.strip()
    sbt_list.append(line)
    print(line)
sbt_dictionary = {value: index + 1 for index, value in enumerate(sbt_list)}
pickle.dump(sbt_dictionary,open("DeepCom_JAVA/vocab.tokens.pkl", 'wb'))