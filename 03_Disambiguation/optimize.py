import random
import os

for i in range(50):
    
    flags = ""
    for j in range(14):
        flags += str(random.randint(0,1)) + " "
    
    os.system("cat ../UD_Portuguese-GSD/pt_gsd-ud-train.conllu | python3 tagger.py -t pt-ud.dat " + flags)
    os.system("cat ../UD_Portuguese-GSD/pt_gsd-ud-test.conllu | python3 tagger.py pt-ud.dat > pt-ud-test.out")
    os.system("python3 ../evaluation_script/conll17_ud_eval.py --verbose ../UD_Portuguese-GSD/pt_gsd-ud-test.conllu pt-ud-test.out")
    
    print("FLAGS: " + str(flags))