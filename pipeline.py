#-*-
from intent_detection_classifier import Intent_Detection_Classifier

from xmlrpc.client import  ServerProxy
from sklearn.metrics import classification_report,precision_recall_fscore_support
from pre_data_deal.data_deal import Intent_Data_Deal
import logging
from configparser import ConfigParser
from intent_re.intent_de.rule_new import intent_detection
import re
import gc


Config=ConfigParser()
Config.read('Config.conf')
HOST=Config['host']['host']
PIPELINE_PORT=int(Config['pipeline']['port'])
LSTM_PORT=int(Config['lstm']['port'])
#CNN_PORT=int(Config['cnn']['port'])
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("pipeline")

idc = Intent_Detection_Classifier('Classifier')
idd=Intent_Data_Deal()
idre=intent_detection()
PATTERN='\\?|？|\\.|。'

# svr_cnn=ServerProxy("http://%s:%s"%(HOST,CNN_PORT))
svr_lstm=ServerProxy('http://127.0.0.1:%s'%LSTM_PORT)



re_change_dict={'bxcp':'康爱保','bzxm':'癌症特别保险金','qj':'蓄意伤害','jb':'感冒','bxzl':'重疾险'}
classifier_dict={'BLSTM':1.4,'CNN':0.4,'NB':0.4,'SVM':0.4,'KNN':0.4,'LR':0.2,'RF':0.4,'DT':0.4,'RE':0.8}

class Pipeline(object):

    def __init__(self):
        self.faq_dict=self.get_FAQ()


    def get_FAQ(self):

        FAQ_dict={}
        with open('./标准FAQ问答.txt','r') as fr:
            for line in fr.readlines():
                line=re.subn(PATTERN,'',line.replace('\n',''))[0].replace('\t\t','\t')
                try:
                    sent=line.split('\t')[0]
                    label=line.split('\t')[1].strip().split(' ')[0]
                    if label not in ['',' ']:
                        FAQ_dict[sent] = label
                except Exception as ex:
                    print(ex,[line])

        return FAQ_dict



    def get_dl_res(self,dl_out):
        ss=[]
        for ele in dl_out:
            s=[e[0] for e in ele]
            ss.append(s)

        # res=[e[0][0] for e in dl_out]
        # ss=res
        return ss
    def get_re_res(self,sents):

        ss=[]
        for sent in sents:
            _ss=[]
            for e in sent.split(' '):
                if e in re_change_dict:
                    _ss.append(re_change_dict[e])
                else:
                    _ss.append(e)
            sent="".join(_ss)
            res=idre.intent_class(sent)
            ss.append([e[0] for e in res])
        return ss

    def _confusion_(self,pre_label,true_label):
        class_re = classification_report(true_label, pre_label)
        # print(class_re)

    def get_result(self,sents,sent_dl,true_label=None):

        sents=[idd.deal_sent(e) for e in sents]
        class_res=idc.infer(sents,['SVM','RF','DT'])
        _logger.info('sklearn classifier finish')


        # res_cnn = self.get_dl_res(svr_cnn.intent(sent_dl))
        res_lstm = self.get_dl_res(svr_lstm.intent(sent_dl))
        res_re=self.get_re_res(sents)
        self._confusion_(class_res['SVM'],true_label)

        class_res['RE']=res_re
        # class_res['CNN']=res_cnn
        class_res['BLSTM']=res_lstm

        result=self.vote(class_res)
        result_dict={}
        if len(sents)!=len(result) or len(res_lstm)!=len(result):
            _logger.error('输入长度和输出长度不一致！')
        else:
            pass
        del class_res
        del res_lstm
        del res_re
        gc.collect()

        return result

    def RE_rule(self,RE_data):
        '''
        正则规则 符合正则规则直接输出正则内容
        :param RE_data:
        :return:
        '''
        re_pattern_0=['保障项目如何申请理赔','得了某类疾病怎样申请理赔','得了某个疾病怎样申请理赔']
        if sum([1 for e in RE_data if e in re_pattern_0])==len(re_pattern_0):
            return RE_data
        else:
            return None

    def vote(self, class_result):
        '''
        投票
        :param class_result:
        :return:
        '''

        use_weight = True
        blstm_list=class_result['BLSTM']
        if use_weight:
            ss = []
            for k, v in dict(class_result).items():
                ele=[(e, classifier_dict[k]) for e in v]
                ss.append(ele)
            num_=len(ss[0])
            result=[]
            for i in range(num_):
                ss_i_dict={}
                for j in range(len(ss)):
                    if isinstance(ss[j][i][0],str):
                        if ss[j][i][0].lower() not in ss_i_dict:
                            ss_i_dict[ss[j][i][0].lower()]=ss[j][i][1]
                        else:
                            num=ss_i_dict[ss[j][i][0].lower()]
                            num+=ss[j][i][1]
                            ss_i_dict[ss[j][i][0].lower()]=num
                    else:
                        for ele in ss[j][i][0]:
                            if ele.lower() not in ss_i_dict:
                                ss_i_dict[ele.lower()]=ss[j][i][1]
                            else:
                                num=ss_i_dict[ele.lower()]
                                num+=ss[j][i][1]
                                ss_i_dict[ele.lower()]=num

                re_data=self.RE_rule(class_result['RE'][i])
                if re_data:
                    result.append(re_data)
                else:
                    ss_sort=[[k,v] for k,v in ss_i_dict.items() if k not in ['',' ']]
                    ss_sort.sort(key=lambda x:x[1],reverse=True)
                    fin_res=ss_sort[0][0]
                    result.append(fin_res)
            return result
        else:
            ss = []
            for k, v in dict(class_result).items():
                ss.append(v)
            if len(set([len(e) for e in ss])) > 1:
                _logger.info('输出长度错误%s' % (' '.join([str(len(e)) for e in ss])))
            else:
                value_num = len(ss[0])
                result = []
                for i in range(value_num):
                    ss_i = []
                    for e in ss:
                        ss_i.append(e[i])

                    sort_rr = []
                    for ele in set(ss_i):
                        sort_rr.append([ele, ss_i.count(ele)])
                    sort_rr.sort(key=lambda x: x[1], reverse=True)
                    result.append(sort_rr[0][0])
                return result


    def get_sent_result(self,sents):

        _sents = [idd.deal_sent(e) for e in sents]
        print(_sents)
        class_res = idc.infer(_sents, ['SVM', 'RF', 'DT'])
        _logger.info('sklearn classifier finish')


        # res_cnn = self.get_dl_res(svr_cnn.intent(sents))
        res_lstm = self.get_dl_res(svr_lstm.intent(_sents))
        res_re = self.get_re_res(_sents)

        class_res['RE'] = res_re
        class_res['BLSTM'] = res_lstm
        result = self.vote(class_res)
        result_dict = {}
        if len(_sents) != len(result) or len(result)!= len(res_lstm):
            _logger.error('输入长度和输出长度不一致！')
        else:
            for line, label in zip(_sents, result):
                result_dict[line] = [label]

        fin_result=[]
        for sent,model_label in zip(sents,result):
            sent=re.subn(PATTERN,'',sent)[0]
            if sent in self.faq_dict:
                fin_result.append(self.faq_dict[sent])
            else:
                fin_result.append(model_label)
        del result
        gc.collect()
        return fin_result






if __name__ == '__main__':

    pipeline=Pipeline()


    #
    # sents=[]
    # sents_dl=[]
    # intents=[]
    # with open('./dataset/train_out_char.txt','r')as fr:
    #     for ele in fr.readlines():
    #         ele=ele.replace('\n','')
    #         sent=ele.split('\t')[0].replace('bos','').replace('eos','').strip()
    #         label=ele.split('\t')[2].strip()
    #         sents_dl.append(' '.join([e for e in sent.split(' ')]))
    #         sent=''.join([e for e in sent.split(' ')])
    #         sents.append(sent)
    #         intent=label.split(' ')[0]
    #         intents.append(intent)
    #
    # result=pipeline.get_result(sents,sents_dl,intents)
    # class_re = classification_report(intents, result)
    # print(class_re)
    # for sent,true_label,label in zip(sents,intents,result):
    #     print(sent,true_label,label)
    #
    sent=['感冒是不是重大疾病','感冒保不保']

    result=pipeline.get_sent_result(sent)

    # print(result)


