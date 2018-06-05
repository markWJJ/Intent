import re
import random
import os
PATH1=os.path.split(os.path.realpath(__file__))[0]
PATH=os.path.split(PATH1)[0]+'/intent_re/intent_de'
import jieba
type_list=['Baozhangxiangmu', 'Jibing', 'Qingjing', 'Wenjian', 'Time', 'Jianejiaoqing', 'Jiaofeinianqi',
           'Shiyi', 'Baoxianzhonglei', 'Baoxianchanpin', 'Fenzhijigou', 'Didian', 'Yiyuan', 'Jiaofeifangshi',
           'Jine', 'Yiyuandengji', 'Baoxianjin', 'Jibingzhonglei', 'Hetonghuifu', 'Baodanjiekuan', 'Mianpeie']
for ele in type_list:
    jieba.load_userdict(PATH + '/data/%s.txt'%ele)

jb=[e.replace('\n','') for e in open(PATH+'/data/Jibing.txt','r')] #疾病
bzxm=[e.replace('\n','') for e in open(PATH+'/data/Baozhangxiangmu.txt','r')] #保障项目
bxzl=[e.replace('\n','') for e in open(PATH+'/data/Baoxianzhonglei.txt','r')] #保险种类
bxcp=[e.replace('\n','') for e in open(PATH+'/data/Baoxianchanpin.txt','r')] #保险产品
jbzl=[e.replace('\n','') for e in open(PATH+'/data/Jibingzhonglei.txt','r')] #疾病种类
qj=[e.replace('\n','') for e in open(PATH+'/data/Qingjing.txt','r')] #情景
fwxm=[e.replace('\n','') for e in open(PATH+'/data/fwxm.txt','r')] #服务项目
dd=[e.replace('\n','') for e in open(PATH+'/data/Didian.txt','r')] #地点
yy=[e.replace('\n','') for e in open(PATH+'/data/Yiyuan.txt','r')] #医院
jffs=[e.replace('\n','') for e in open(PATH+'/data/Jiaofeifangshi.txt','r')] #缴费方式
yydj=[e.replace('\n','') for e in open(PATH+'/data/Yiyuandengji.txt','r')] #医院等级
bxj=[e.replace('\n','') for e in open(PATH+'/data/Baoxianjin.txt','r')] #保险金
sy=[e.replace('\n','') for e in open(PATH+'/data/Shiyi.txt','r')] #释义

entity_list=[]
entity_list.extend(jb)
entity_list.extend(bzxm)
entity_list.extend(bxzl)
entity_list.extend(bxcp)
entity_list.extend(jbzl)
entity_list.extend(sy)



class Intent_Data_Deal(object):

    def __init__(self):

        pass
    def deal_sent(self,line):
        '''
        对句子进行处理
        :param sent:
        :return:
        '''
        pattern = '\d{1,3}(\\.|，|、|？|《|》)|《|》|？|。'
        line = line.replace('\n', '').replace('Other','other')
        sent=line
        sent = re.subn(pattern, '', sent)[0]
        ss = []
        if sent in sy:
            ss.append('sy')
        else:
            sents=[e for e in jieba.cut(sent)]
            for e in sents :
                if e in jb:
                    ss.append('jb')
                elif e in bzxm:
                    ss.append('bzxm')
                elif e in bxzl:
                    ss.append('bxzl')
                elif e in bxcp:
                    ss.append('bxcp')
                elif e in qj:
                    ss.append('qj')
                elif e in bxj:
                    ss.append('bxj')
                elif e in fwxm:
                    ss.append('fwxm')
                elif e in jbzl:
                    ss.append('jbzl')
                else:
                    ss.append(e)

        sent = []
        for word in ss:
            word = word.lower()
            if word not in ['bzxm', 'jb', 'qj', 'bxj', 'bxcp', 'eos', 'bos', 'bxzl', 'sy', 'fwxm', 'bqx', 'jbzl']:
                s = [e for e in word]
                sent.extend(s)
            else:
                sent.append(word)
        sent=' '.join(sent)

        return sent


    def deal_sent_file(self,line):
        '''
        对句子进行处理
        :param sent:
        :return:
        '''
        pattern = '\d{1,3}(\\.|，|、|？)|《|》|？|。'
        line = line.replace('\n', '').strip()
        sent=''
        ll=''
        line=line.replace('\t\t','\t').replace('Other','other')
        if '\t' in line:

            ll=str(line).split('\t')[1].strip()
            sent=str(line).split('\t')[0]
        else:
            try:
                ll=str(line).split(' ')[1].strip()
                sent=str(line).split(' ')[0]
            except:
                print([line])

        sent = re.subn(pattern, '', sent)[0]
        ss = []
        if sent in sy:
            ss.append('sy')
        else:
            sents=[e for e in jieba.cut(sent)]
            for id,e in enumerate(sents):
                if e not in []:
                    if e in jb:
                        ss.append('jb')
                    elif e in bzxm:
                        ss.append('bzxm')
                    elif e in bxzl :
                        ss.append('bxzl')
                    elif e in bxcp:
                        ss.append('bxcp')
                    elif e in qj:
                        ss.append('qj')
                    elif e in bxj:
                        ss.append('bxj')
                    elif e in fwxm:
                        ss.append('fwxm')
                    elif e in jbzl:
                        ss.append('jbzl')
                    else:
                        ss.append(e)
        sent=' '.join(ss)
        label = ll
        sent = 'BOS' + ' ' + sent + ' ' + 'EOS'
        entity = ' '.join(['O'] * len(sent.split(' ')))
        res = sent + '\t' + entity + '\t' + label

        sent = res.split('\t')[0]
        try:
            label = res.split('\t')[2]
        except:
            print(res)

        sent = sent.split(' ')
        ss = []
        for word in sent:
            word = word.lower()
            if word not in ['bzxm', 'jb', 'qj', 'bxj', 'bxcp', 'eos', 'bos', 'bxzl', 'sy', 'fwxm', 'bqx','jbzl']:
                s = [e for e in word]
                ss.extend(s)
            else:
                ss.append(word)

        sents = ss
        slot = ['o'] * len(sents)

        sent = ' '.join(sents)
        slot = ' '.join(slot)

        return sent+'\t'+slot+'\t'+label

    def deal_file(self,input_file_name,train_file_name,dev_file_name,split_rate=0.2):
        '''
        将输入的标注数据 转换为带实体标签的char数据
        :param file_name:
        :return:
        '''
        fw_train=open(train_file_name,'w')
        fw_dev=open(dev_file_name,'w')
        num_dict={}
        data=set()
        with  open(input_file_name,'r') as fr:
            for ele in fr.readlines():
                e=self.deal_sent_file(ele)
                data.add(e)

            fr.close()
        data=list(data)
        random.shuffle(data)
        split_num=int(len(data)*split_rate)

        # dev write
        for ele in data[:split_num]:
            fw_dev.write(ele)
            fw_dev.write('\n')

            label=ele.replace('\n','').split('\t')[2]
            labels=label.split(' ')
            for ee in labels:
                ee=ee.lower()
                if ee not in num_dict:
                    num_dict[ee]=1
                else:
                    s=num_dict[ee]
                    s+=1
                    num_dict[ee]=s

        # train write

        for ele in data[split_num:]:
            fw_train.write(ele)
            fw_train.write('\n')

            label=ele.replace('\n','').split('\t')[2]
            labels=label.split(' ')
            for ee in labels:
                ee=ee.lower()
                if ee not in num_dict:
                    num_dict[ee]=1
                else:
                    s=num_dict[ee]
                    s+=1
                    num_dict[ee]=s
        print('label num dict')
        print(num_dict)
if __name__ == '__main__':


    idd=Intent_Data_Deal()
    # print(idd.deal_sent('康爱保有什么保险责任吗'))
    idd.deal_file('意图识别数据_all.txt','../dataset/train_out_char.txt','../dataset/dev_out_char.txt')
