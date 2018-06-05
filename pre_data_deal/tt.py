
import re
pattern='。|，|？|\\?'
ss=[]
ss_=[]

sent_dict={}
fw=open('整理标准_去重.txt','w')
for ele in open('./整理标准.txt','r'):
    ele=ele.replace('\t\t','\t').replace('\n','')
    ele=re.subn(pattern,'',ele)[0]
    try:
        sent=ele.split('\t')[0].strip()
        label=ele.split('\t')[1]

        if sent and sent not in ['\n']:
            sent_dict[sent]=label
    except Exception as ex:
        # print([ele])
        pass


for k,v in sent_dict.items():
    fw.write(k)
    fw.write('\t\t')
    fw.write(v)
    fw.write('\n')


fw=open('排序.txt','w')

sort_dict={}

for k,v in sent_dict.items():
    if v not in sort_dict:
        sort_dict[v]=[k]
    else:
        ss=sort_dict[v]
        ss.append(k)
        sort_dict[v]=ss


for k,v in sort_dict.items():
    for ele in v:
        fw.write(ele)
        fw.write('\t\t')
        fw.write(k)
        fw.write('\n')


# ss_dict=[]
# for ele in open('./整理标准_去重.txt','r').readlines():
#     ele=ele.replace('\n','')
#     ele.split('\t')


