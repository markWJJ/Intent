
dd_dict={}

for line in open('./意图识别.txt','r').readlines():
    line=line.replace('\n','')
    intent_new=line.split(':')[1].replace('\t','').strip()
    intent_old=line.split(':')[0].replace('\t','').strip()


    if intent_old not in dd_dict:
        if intent_new:
            dd_dict[intent_old]=intent_new
        else:
            dd_dict[intent_old]=intent_old


intent_dict={}
for line in open('./意图数据_all.txt','r').readlines():
    line=line.replace('\n','').replace('\t\t','\t')
    try:
        sent=line.split('\t')[0]

        intent=line.split('\t')[1]
        ss=[]
        for ele in intent.split(' '):
            if ele in dd_dict:
                ss.append(dd_dict[ele])

        if sent not in intent_dict:
            intent_dict[sent]=ss
    except:
        print(line)

ww=open('./意图识别数据_all.txt','w')

for k,v in intent_dict.items():
    ww.write(k)
    ww.write('\t\t')
    if ' '.join(v):
        ww.write(' '.join(v))
    else:
        ww.write('other')
    ww.write('\n')
