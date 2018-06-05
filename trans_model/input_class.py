import numpy as np
import random

class data_transfer(object):
    def __init__(self,pos_rel,neg_rel):
        self.pos_rel=pos_rel
        self.neg_rel=neg_rel
        self.dictionary={'none':0}
        self.train_data=[]

    def words_dictionary(self):
        num=1
        with open(self.pos_rel,encoding='utf-8') as pos_file:
            for i in pos_file:
                i=i.replace('\n','').split(' ')
                for word in i:
                    if word not in self.dictionary:
                        self.dictionary[word]=num
                        num+=1
        with open(self.neg_rel, encoding='utf-8') as neg_file:
            for i in neg_file:
                i = i.replace('\n', '').split(' ')
                for word in i:
                    if word not in self.dictionary:
                        self.dictionary[word]=num
                        num+=1

    def words_to_id(self,sentence_len):
        with open(self.pos_rel,encoding='utf-8') as pos_file:
            for i in pos_file:
                i = i.replace('\n', '').split(' ')
                if len(i)>=sentence_len:
                    i=i[0:sentence_len]
                else:
                    i.extend(['none']*(sentence_len-len(i)))
                temp_list=[]
                for word in i:
                    temp_list.extend([self.dictionary[word]])
                temp_list.extend([0])
                self.train_data.append(temp_list)

        with open(self.neg_rel,encoding='utf-8') as neg_file:
            for i in neg_file:
                i = i.replace('\n', '').split(' ')
                if len(i)>=sentence_len:
                    i=i[0:sentence_len]
                else:
                    i.extend(['none']*(sentence_len-len(i)))
                temp_list=[]
                for word in i:
                    temp_list.extend([self.dictionary[word]])
                temp_list.extend([1])
                self.train_data.append(temp_list)
        random.shuffle(self.train_data)
        return self.train_data

    def train_dev_split(self,data_list,split_rate=0.8):
        length=len(data_list)
        train_length=int(split_rate*length)
        train_data=data_list[:train_length]
        dev_data=data_list[train_length:]
        return train_data,dev_data

if __name__ == '__main__':
    data_transfer=data_transfer('pos.txt','neg.txt')
    data_transfer.words_dictionary()
    train_data=data_transfer.words_to_id(30)
    train_data,dev_data=data_transfer.train_dev_split(train_data,0.8)
    print(train_data)

