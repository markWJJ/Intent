import numpy as np
import re

positive_path='data/task3&4_pos.txt'
negative_path='data/task3&4_neg.txt'

pos_file=open('data/pos.txt','w',encoding='utf-8')
neg_file=open('data/neg.txt','w',encoding='utf-8')

pattern='！|。|~|,|\\.|，|#|…|//|？|“|”|!|～|、|；'

with open(positive_path,'r',encoding='utf-8') as positive_file:
    for i in positive_file:
        i=i.replace('\n','')
        if i.startswith('<'):
            pass
        else:
            sent=re.subn(pattern,'',i)
            sent = ' '.join([e for e in sent[0]])
            pos_file.write(sent)
            pos_file.write('\n')

with open(negative_path,'r',encoding='utf-8') as negative_file:
    for i in negative_file:
        i=i.replace('\n','')
        if i.startswith('<'):
            pass
        else:
            sent=re.subn(pattern,'',i)
            sent=' '.join([e for e in sent[0]])
            neg_file.write(sent)
            neg_file.write('\n')



