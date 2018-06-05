from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import svm,neighbors,tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys,getopt
import os
import numpy as np
from intent_detection import Intent_Detection
from sklearn.metrics import classification_report,precision_recall_fscore_support
import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.externals import joblib
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("intent_classifiers")
jieba.load_userdict('./user_dict.txt')
classifiers=['NB','SVM','KNN','LR','RF','DT']



class Intent_Detection_Classifier(Intent_Detection):

    def __init__(self,class_name):
        self.class_name=class_name
        self.id2intent={}
        self.id2sent={}
        if os.path.exists('./classifier_model'):

            self.count_vect = joblib.load('./classifier_model/count_vect.pkl')
            self.NB_clf = joblib.load('./classifier_model/NB_model.pkl')
            self.SVM_clf = joblib.load('./classifier_model/SVM_model.pkl')
            self.KNN_clf = joblib.load('./classifier_model/KNN_model.pkl')
            self.LR_clf = joblib.load('./classifier_model/LR_model.pkl')
            self.RF_clf = joblib.load('./classifier_model/RF_model.pkl')
            self.DT_clf = joblib.load('./classifier_model/DT_model.pkl')
        else:
            os.mkdir('./classifier_model/')


    def pre_data(self,train_file,dev_file,max_len=10):

        train_data_sent=[]
        train_data_label=[]

        dev_data_sent=[]
        dev_data_label=[]
        sent_dict={'none':0}
        label_dict={}
        sent_index=1
        label_index=0

        for line in open(train_file,'r').readlines():
            try:
                line=line.replace('\n','').replace('bos','').replace('eos','').strip()
                sent=str(line.split('\t')[0])
                labels=line.split('\t')[2].strip()
                iter_sent=' '.join([e for e in jieba.cut(''.join(sent.split(' ')))])
                train_data_sent.append(iter_sent)
                train_data_label.append(labels.split(' ')[0])
            except:
                print(line)


        for line in open(dev_file,'r').readlines():
            line = line.replace('\n', '').replace('bos', '').replace('eos', '')
            sent=line.split('\t')[0]
            labels=line.split('\t')[2].strip()
            iter_sent=' '.join([e for e in jieba.cut(''.join(sent.split(' ')))])

            for word in jieba.cut(''.join(sent.split(' '))):
                if word not in sent_dict:
                    sent_dict[word]=sent_index
                    sent_index+=1


            for ll in labels.split(' '):
                if ll not in label_dict:
                    label_dict[ll]=label_index
                    label_index+=1

                dev_data_sent.append(iter_sent)
                dev_data_label.append(ll)

        for k,v in label_dict.items():
            self.id2intent[v]=k

        for k,v in sent_dict.items():
            self.id2sent[v]=k

        return train_data_sent,train_data_label,dev_data_sent,dev_data_label


    def build_model(self,train_file,dev_file):
        if not os.path.exists('./classifier_model'):
            os.mkdir('./classifier_model/')
        train_x,self.train_y,dev_x,self.dev_y=self.pre_data(train_file,dev_file)

        self.count_vec = CountVectorizer()
        self.X_count_train = self.count_vec.fit_transform(train_x)

        self.X_count_dev=self.count_vec.transform(dev_x)
        joblib.dump(self.count_vec, './classifier_model/count_vect.pkl')


    def _train(self,classifier_name):

        clf = GaussianNB()
        if classifier_name=='NB':
            clf = GaussianNB()  # 默认
        elif classifier_name=='SVM':
            clf = svm.LinearSVC(C=2.0)
        elif classifier_name=='KNN':
            clf = neighbors.KNeighborsClassifier()
        elif classifier_name == 'LR':
            clf = LogisticRegression(penalty='l2')
        elif classifier_name=='RF':
            clf = RandomForestClassifier(n_estimators=5)
        elif classifier_name=='DT':
            clf = tree.DecisionTreeClassifier()
        else:
            _logger.error('没有该类别')
        clf.fit(self.X_count_train.toarray(), self.train_y)
        joblib.dump(clf, './classifier_model/%s_model.pkl' % classifier_name)

        predict_train_y = clf.predict(self.X_count_train.toarray())
        p, r, f, s = precision_recall_fscore_support(self.train_y, predict_train_y)
        avg_p = np.mean(p)
        avg_r = np.mean(r)
        avg_f = np.mean(f)
        _logger.info(
            '%s avg_train_precision:%s avg_train_recall:%s avg_train_f:%s' % (classifier_name, avg_p, avg_r, avg_f))

        predict_dev_y = clf.predict(self.X_count_dev.toarray())
        p_dev, r_dev, f_dev, s_dev = precision_recall_fscore_support(self.dev_y, predict_dev_y)
        avg_p_dev = np.mean(p_dev)
        avg_r_dev = np.mean(r_dev)
        avg_f_dev = np.mean(f_dev)
        _logger.info(
            '%s avg_dev_precision:%s avg_dev_recall:%s avg_dev_f:%s' % (
                classifier_name, avg_p_dev, avg_r_dev, avg_f_dev))
        _logger.info('\n')


    def infer(self,sent_list,classifier_name='ALL'):
        '''
        推断模块 如果classifier_name=All
        :param sent_list:
        :param classifier_name: if ALL 使用所有模型
        :return:
        '''


        sents=[]
        for sent in sent_list:
            sent=sent.replace(' ','')
            sents.append(' '.join([e for e in jieba.cut(sent)]))
        sent_vec=self.count_vect.transform(sents)


        if classifier_name=='ALL':
            classifier_names=classifiers
        else:
            classifier_names=classifier_name

        out_dict={}
        for class_name in classifier_names:
            if class_name not in classifiers:
                _logger.error('没有该分类方法')
            elif class_name=='NB':
                predict_out=self.NB_clf.predict(sent_vec.toarray())
                out_dict[class_name]=predict_out
            elif class_name=='SVM':
                predict_out = self.SVM_clf.predict(sent_vec.toarray())
                out_dict[class_name] = predict_out
            elif class_name=='KNN':
                predict_out = self.KNN_clf.predict(sent_vec.toarray())
                out_dict[class_name] = predict_out
            elif class_name=='LR':
                predict_out = self.LR_clf.predict(sent_vec.toarray())
                out_dict[class_name] = predict_out
            elif class_name=='RF':
                predict_out = self.RF_clf.predict(sent_vec.toarray())
                out_dict[class_name] = predict_out
            elif class_name=='DT':
                predict_out = self.DT_clf.predict(sent_vec.toarray())
                out_dict[class_name] = predict_out

        return out_dict


    def train(self,classifier_names):
        '''
        训练模块
        :param classifier_names:
        :return:
        '''
        for ele in classifiers:
            _logger.info('is train %s'%ele)
            self._train(ele)
#


if __name__ == '__main__':
    idb=Intent_Detection_Classifier('Classifier')
    idb.build_model('./dataset/train_out_char.txt','./dataset/dev_out_char.txt')
    idb.train(classifiers)
    sent=['bxcp有保额限制吗？','你们有什么产品推荐吗']
    res=idb.infer(sent)
    print(res)
