import re
import logging
import jieba
import os
from intent_sys import Tree
import xlrd
PATH=os.path.split(os.path.realpath(__file__))[0]
type_list=['Baozhangxiangmu', 'Jibing', 'Qingjing', 'Wenjian', 'Time', 'Jianejiaoqing', 'Jiaofeinianqi',
           'Shiyi', 'Baoxianzhonglei', 'Baoxianchanpin', 'Fenzhijigou', 'Didian', 'Yiyuan', 'Jiaofeifangshi',
           'Jine', 'Yiyuandengji', 'Baoxianjin', 'Jibingzhonglei', 'Hetonghuifu', 'Baodanjiekuan', 'Mianpeie']
for ele in type_list:
    jieba.load_userdict(PATH + '/data/%s.txt'%ele)
jieba.load_userdict(PATH+'/data/gsmz.txt')
#jieba.load_userdict(PATH + '/data/user_dict.txt')
# jieba.load_userdict(PATH+'/data/user_dict.txt')
# jieba.load_userdict(PATH+'/data/bzxm.txt')
# jieba.load_userdict(PATH+'/data/jb.txt')
# jieba.load_userdict(PATH+'/data/st.txt')
# jieba.load_userdict(PATH+'/data/tjxm.txt')
# jieba.load_userdict(PATH+'/data/pt.txt')
# jieba.load_userdict(PATH+'/data/bxcp.txt')
# jieba.load_userdict(PATH+'/data/gs.txt')
# jieba.load_userdict(PATH+'/data/tsrq.txt')
# jieba.load_userdict(PATH+'/data/jbzl.txt')
# jieba.load_userdict(PATH+'/data/qj.txt')
# jieba.load_userdict(PATH+'/data/jbhz.txt')
# jieba.load_userdict(PATH+'/data/fwxm.txt')



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data")


jb=[e.replace('\n','') for e in open(PATH+'/data/Jibing.txt','r')] #疾病
bzxm=[e.replace('\n','') for e in open(PATH+'/data/Baozhangxiangmu.txt','r')] #保障项目
bxzl=[e.replace('\n','') for e in open(PATH+'/data/Baoxianzhonglei.txt','r')] #保险种类
st=[e.replace('\n','') for e in open(PATH+'/data/st.txt','r')] #身体部位
tjxm=[e.replace('\n','') for e in open(PATH+'/data/tjxm.txt','r')] #体检项目
pt=[e.replace('\n','') for e in open(PATH+'/data/pt.txt','r')] #平台
bxcp=[e.replace('\n','') for e in open(PATH+'/data/Baoxianchanpin.txt','r')] #保险产品
gs=[e.replace('\n','') for e in open(PATH+'/data/gs.txt','r')] #公司名
tsrq=[e.replace('\n','') for e in open(PATH+'/data/tsrq.txt','r')] #特殊人群
jbzl=[e.replace('\n','') for e in open(PATH+'/data/Jibingzhonglei.txt','r')] #疾病种类
qj=[e.replace('\n','') for e in open(PATH+'/data/Qingjing.txt','r')] #情景
jbhz=[e.replace('\n','') for e in open(PATH+'/data/jbhz.txt','r')] #疾病患者
fwxm=[e.replace('\n','') for e in open(PATH+'/data/fwxm.txt','r')] #服务项目
dd=[e.replace('\n','') for e in open(PATH+'/data/Didian.txt','r')] #地点
yy=[e.replace('\n','') for e in open(PATH+'/data/Yiyuan.txt','r')] #医院
jffs=[e.replace('\n','') for e in open(PATH+'/data/Jiaofeifangshi.txt','r')] #缴费方式
yydj=[e.replace('\n','') for e in open(PATH+'/data/Yiyuandengji.txt','r')] #医院等级
bxj=[e.replace('\n','') for e in open(PATH+'/data/Baoxianjin.txt','r')] #保险金
sy=[e.replace('\n','') for e in open(PATH+'/data/Shiyi.txt','r')] #释义
gsmc=[e.replace('\n','') for e in open(PATH+'/data/gsmz.txt','r')] #公司名称




label_dict={'D1':'介绍','D2': '问询', 'D3':'变更','D4': '申请','D5': '区别','D6': '限制','D7': '包含','D8': '是否',
            'D11':'合同介绍'}


#冲突列表

TREE = Tree(PATH+'/data/意图识别.txt')


class intent_detection(object):

    def __init__(self):
        self.id_list=[]
        self.jbs = '|'.join(jb)
        self.bxcps = '|'.join(bxcp)
        self.bzxms = '|'.join(bzxm)
        self.bxzls = '|'.join(bxzl)
        self.tsrqs = '|'.join(tsrq)
        self.jbzl = '|'.join(jbzl)
        self.qjs = '|'.join(qj)
        self.jbhzs = '|'.join(jbhz)
        self.sts = '|'.join(st)
        self.tjxms = '|'.join(tjxm)
        self.fwxms='|'.join(fwxm)
        self.pts='|'.join(pt)
        self.sy='|'.join(sy)
        self.bxjs='|'.join(bxj)
        self.yydjs='|'.join(yydj)
        self.jffss='|'.join(jffs)
        self.yys='|'.join(yy)
        self.dds='|'.join(dd)
        self.gsmc='|'.join(gsmc)


    def conflit(self,label_list):
        '''
        解决冲突
        :param label_list:
        :return:
        '''
        if len(label_list)==1:
            return label_list
        else:
            label_list=list(label_list)
            label_list=TREE.conflit_deal(label_list)
            return label_list


    def intent_class(self,sent:str):
        '''
        意图分类功能实现
        :param sent:
        :return:
        '''

        label='Other'
        label_list=[]
        lev_1=self.level_1(sent)
        if lev_1=='D1': #介绍类
            ss=self.level_2_1(sent)

        if lev_1=='D2': # 问询类
            ss=self.level_2_2(sent)
            print(ss,'\t',sent)

    def level_1(self,sent):
        '''
        第一层体系
        :param sent:
        :return:
        '''
        flag=False
        pattern_1='什么意思|解释一下|是啥|定义|什么是|含义|不知道是啥|干什么用的|有什么用|啥意思|是指什么|介绍|构成'
        if re.search(pattern_1,sent):
            flag=True
            return 'D1'

        pattern_2='更改|变更'
        if re.search(pattern_2,sent):
            flag=True
            # print('变更',sent)

        pattern_3='申请'
        if re.search(pattern_3,sent):
            flag=True
            # print('申请',sent)

        pattern_4='区别'
        if re.search(pattern_4, sent):
            flag=True
            # print('区别', sent)

        pattern_5 = '限制|要求'
        if re.search(pattern_5, sent):
            flag=True
            # print('限制', sent)

        pattern_6 = '包含|有哪些'
        if re.search(pattern_6, sent):
            flag=True
            # print('包含', sent)

        pattern_7 = '是否|是不是'
        if re.search(pattern_7, sent):
            flag=True
            # print('是否', sent)

        if not flag:
            # print('问询',sent)
            return 'D2'

    def level_2_1(self,sent):
        '''
        介绍 层级细分
        :param sent:
        :return:
        '''
        pattern='合同' #合同介绍
        pattern_1='电子合同' #电子合同介绍
        pattern_2='合同'#询问合同的构成
        if re.search(pattern,sent):
            if re.search(pattern_1,sent):
                return '电子合同介绍'
            else:
                return '询问合同的构成'

    def level_2_2(self,sent):
        '''
        问询 层级细分
        :param sent:
        :return:
        '''
        # 合同相关问询
        pattern='合同'

        # 询问合同恢复
        pattern_1 = '复效合同|合同效力恢复|合同恢复|怎么恢复合同|如何恢复合同|恢复合同流程|怎样才能恢复合同|想恢复合同|恢复合同需要什么|可以恢复合同吗|恢复合同需要做什么|恢复合同需要提供什么|什么情况下可以恢复合同|什么时候合同效力恢复|合同恢复是什么'

        #询问复效期
        pattern_2 = '合同效力恢复|复效期'

        #复效期有无收取滞纳金
        pattern_2_1 = '复效期.*滞纳金' #复效期有无收取滞纳金

        #询问合同终止
        pattern_3='合同终止|终止合同|结束合同|合同.*终止'

        #合同自动终止的情景
        pattern_3_1='信托合同.*自动终止'

        #合同丢失
        pattern_4='合同丢失|合同不见.*怎么办|合同找不到'

        #询问合同领取方式
        pattern_5='领?[取|拿]|获取'

        #询问合同种类
        pattern_6 = '合同.*属于.*类型|合同种类|什么类型.*合同|属于.*类型.*合同|合同保什么|合同.*有.*意义|合同.*有.*作?用' \
                  '|合同.*有.*帮助|保险合同有.*类|保险合同分类'

        #:合同相关
        if re.search(pattern,sent):
            flag=False
            if re.search(pattern_1,sent) :
                flag=True
                return '询问合同恢复'

            if re.search(pattern_2,sent):
                flag=True
                if re.search(pattern_2_1,sent):
                    return '复效期有无收取滞纳金'
                else:
                    return '询问复效期'

            if re.search(pattern_3,sent):
                flag=True
                if re.search(pattern_3_1,sent):
                    return '合同自动终止的情景'
                else:
                    return '询问合同终止'

            if re.search(pattern_4,sent):
                flag=True
                return '合同丢失'

            if re.search(pattern_5,sent):
                flag=True
                return '询问合同领取方式'

            if re.search(pattern_6,sent):
                flag=True
                return '询问合同种类'

            if not flag:
                return ':合同相关'

        #:公司相关
        pattern_7='.*'

        #公司地址
        pattern_8='公司地址|客服中心.*在.*地方|客服中心地址'

        #询问联系方式
        pattern_9='客服热线|(客服|公司)电话|云助理电话|联系.*(方式|方法)'

        #询问分公司联系方式
        pattern_9_1='分公司|分公司|分支机构'

        #公司相关
        if re.search(pattern_7,sent):
            if re.search(pattern_8,sent):
                return '公司地址'
            if re.search(pattern_9,sent):
                if re.search(pattern_9_1,sent):
                    return '询问分公司联系方式'
                else:
                    return '询问联系方式'







if __name__ == '__main__':

    id=intent_detection()



    filer=open('./FAQ_1.txt','r')
    fw=open('./writeb1.txt','w')
    res=[]
    for num,line in enumerate(filer.readlines()):
        if line not in ['','\n']:
            labels=id.intent_class(line)
            ss=[]
            # for label in labels:
            #     ss.append(label[0])
            # new_label=" ".join(ss)
            # print(new_label,'\t\t',line)
            # fw.write(new_label)
            # fw.write('\t\t')
            # fw.write(line.replace('\n',''))
            # fw.write('\n')

    # while True:
    #     text=input('输入')
    #     print(id.intent_class(text))
