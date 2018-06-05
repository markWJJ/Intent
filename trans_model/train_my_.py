'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import sys
sys.path.append('../')
from focal_loss import focal_loss
import tensorflow as tf
from data_preprocess import Intent_Slot_Data
from trans_model.modules import *
import random
import os, codecs
import time
from trans_model.input_class import data_transfer
from sklearn.metrics import classification_report,precision_recall_fscore_support
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")


class Config:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 16  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 30  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.9
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


config=Config

class Graph():
    def __init__(self,vocab_num):
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        self.x = tf.placeholder(tf.int32, shape=(None, config.maxlen))
        self.intent = tf.placeholder(shape=(None,), dtype=tf.int32)
        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                  vocab_size=vocab_num,
                                  num_units=config.hidden_units,
                                  scale=False,
                                  scope="enc_embed")

            # embed=tf.Variable(tf.random_normal(shape=(vocab_num,config.hidden_units),dtype=tf.float32))
            # self.enc=tf.nn.embedding_lookup(embed,self.x)


            # # Positional Encoding
            # if config.sinusoid:
            #     self.enc += positional_encoding(self.x,
            #                       num_units=config.hidden_units,
            #                       zero_pad=False,
            #                       scale=False,
            #                       scope="enc_pe")
            # else:
            #     self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
            #                       vocab_size=config.maxlen,
            #                       num_units=config.hidden_units,
            #                       zero_pad=False,
            #                       scale=False,
            #                       scope="enc_pe")


            ## Dropout
            # self.enc = tf.layers.dropout(self.enc,
            #                             rate=config.dropout_rate,
            #                             training=True)
            #
            # # Blocks
            # for i in range(2):
            #     with tf.variable_scope("num_blocks_{}".format(i)):
            #         ### Multihead Attention
            #         self.enc = multihead_attention(queries=self.enc,
            #                                         keys=self.enc,
            #                                         num_units=config.hidden_units,
            #                                         num_heads=config.num_heads,
            #                                         dropout_rate=config.dropout_rate,
            #                                         is_training=True,
            #                                         causality=False)
            #
            #         ### Feed Forward
            #         self.enc = feedforward(self.enc, num_units=[4*config.hidden_units, config.hidden_units])


            # enc=tf.expand_dims(self.enc,3)
            # encoder=tf.nn.max_pool(enc, ksize = [1,config.maxlen, 1, 1], strides = [1, config.maxlen, 1, 1], padding = 'VALID', name = 'maxpool1')

            encoder=tf.reduce_mean(self.enc,1)
            # encoder=tf.layers.dense(encoder,512,activation=tf.nn.tanh)
            # print(encoder)

            # lstm=tf.contrib.rnn.BasicLSTMCell(config.hidden_units)
            # encs=tf.unstack(self.enc,config.maxlen,1)
            #
            # encoder,_=tf.nn.static_rnn(lstm,encs,dtype=tf.float32)
            # encoder=encoder[0]

            # filter_w_1 = tf.Variable(tf.truncated_normal(shape=(3, config.hidden_units, 1, 512), stddev=0.1), name="W")
            # filter_b_1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            #
            # cnn_out1 = tf.nn.conv2d(enc, filter_w_1, strides=[1, 1, 1, 1], padding='VALID')  #
            # cnn_out1 = tf.nn.relu(tf.nn.bias_add(cnn_out1, filter_b_1))
            # cnn_out1 = tf.nn.max_pool(cnn_out1, ksize=[1, config.maxlen-3 + 1, 1, 1], strides=[1, 1, 1, 1],
            #                           padding='VALID')

            # encoder=tf.reshape(encoder,[-1,512])
            # encoder=tf.transpose(self.enc,[1,0,2])[0]


            # print(encoder)
            #
            # encoder=tf.reshape(self.enc,[-1,30*512])
            # encoder=tf.layers.dense(encoder,512,activation=tf.nn.tanh)

            # rate=0.8
            # num=int(config.maxlen*rate)
            # encs=tf.unstack(self.enc,num=config.maxlen,axis=1)
            # enc_list=[]
            # ss=list(range(config.maxlen))
            # random.shuffle(ss)
            # for i in ss[:25]:
            #     enc_list.append(encs[i])
            #
            # enc_list=tf.stack(enc_list,1)
            # encoder=tf.reduce_mean(enc_list,1)
            #
            # # w=tf.Variable(tf.random_normal(shape=(512,512)))
            # # encoder=tf.matmul(encoder,w)
            # encoder=tf.layers.dense(encoder,512,activation=tf.nn.relu,use_bias=True)
            # # encoder=tf.layers.dense(encoder,512,activation=tf.nn.relu,use_bias=True)
            #
            # # print(encoder)
            #
            #
            # lstm_w = tf.Variable(tf.random_normal(shape=(512, 2), dtype=tf.float32))
            # lstm_b = tf.Variable(tf.random_normal(shape=(2,), dtype=tf.float32))
            # logit = tf.add(tf.matmul(encoder, lstm_w), lstm_b)
            # self.soft_logit = tf.nn.softmax(logit, 1)
            # l2_loss = tf.get_collection('l2')
            # # intent=label_smoothing(tf.cast(tf.one_hot(self.intent,2),tf.float32))
            # intent=tf.cast(tf.one_hot(self.intent,2,1,0),tf.float32)
            #
            # intent_loss=-tf.reduce_sum(intent*tf.log(tf.clip_by_value(self.soft_logit, 1e-10, 1.0, name=None)))

            # intent_loss = tf.losses.softmax_cross_entropy(intent, logit, reduction=tf.losses.Reduction.NONE)
            # intent_loss = intent_loss
            # intent_loss = tf.reduce_mean(intent_loss)
            self.label_embeding=tf.Variable(tf.random_normal(shape=(2,512),dtype=tf.float32))
            self.sent_sim_emb=encoder
            sent_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(encoder), axis=1))
            intent_norms = tf.unstack(self.label_embeding, 2, 0)
            cosins = []
            # 内积
            for ele in intent_norms:
                intent_norm = tf.sqrt(tf.reduce_sum(tf.square(ele)))
                ele = tf.expand_dims(ele, -1)
                sent_intent = tf.matmul(encoder, ele)
                sent_intent=tf.reshape(sent_intent,[-1,])
                cosin = sent_intent / (sent_emb_norm * intent_norm)
                cosins.append(cosin)
            cosin = tf.stack(cosins, 1)
            self.soft_logit = tf.nn.softmax(cosin, 1)
            l2_loss = tf.get_collection('l2')
            # intent=label_smoothing(tf.cast(tf.one_hot(self.intent,2),tf.float32))
            intent=tf.cast(tf.one_hot(self.intent,2,1,0),tf.float32)

            intent_loss=focal_loss(self.soft_logit,intent)

            # intent_loss=-tf.reduce_sum(intent*tf.log(tf.clip_by_value(self.soft_logit, 1e-10, 1.0, name=None)))

            # intent_loss = tf.losses.softmax_cross_entropy(intent, cosin, reduction=tf.losses.Reduction.NONE)
            # intent_loss = intent_loss
            # intent_loss = tf.reduce_mean(intent_loss)


            self.loss_op = intent_loss
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.4).minimize(self.loss_op)

            # 梯度截断
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=0.4)
            grads_vars = self.opt.compute_gradients(self.loss_op)
            capped_grads_vars = [[tf.clip_by_value(g, -1e-5, 1.0), v]
                                 for g, v in grads_vars]
            self.optimizer = self.opt.apply_gradients(capped_grads_vars)


    def train(self,train_data,dev_data):


        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )


        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            train_data=np.array(train_data)
            dev_data=np.array(dev_data)
            train_sent=np.array(train_data[:,:-1])
            train_intent=np.array(train_data[:,-1])
            print(train_sent.shape)
            dev_sent=np.array(dev_data[:,:-1])
            dev_intent=np.array(dev_data[:,-1])

            for j in range(100):
                _logger.info('第%s次epoch'%j)
                start_time = time.time()
                # for i in range(num_batch):

                # intent_loss,softmax_logit, _ = sess.run([self.loss_op,self.soft_logit, self.optimizer], feed_dict={self.x: sent,
                #                                                                      self.intent: intent,
                #                                                                      })




                train_softmax_logit, train_loss,_= sess.run([self.soft_logit, self.loss_op,self.optimizer],
                                                       feed_dict={self.x: train_sent,
                                                                  self.intent: train_intent,
                                                                  })
                train_res=classification_report(train_intent,np.argmax(train_softmax_logit,1))
                print('train_loss',train_loss)
                print('train_acc',train_res)

                dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.x: dev_sent,
                                                                 self.intent: dev_intent,
                                                                 })
                print('dev_loss',dev_loss)
                res_dev=classification_report(dev_intent,np.argmax(dev_softmax_logit,1))
                print('dev_acc',res_dev)

                # train_intent_acc = self.intent_acc(train_softmax_logit, train_intent)

                # _logger.info('train_intent_loss:%s train_intent_acc:%s'%(train_loss,train_intent_acc))
                # _logger.info('dev_intent_loss:%s dev_intent_acc:%s'%(dev_loss,dev_intent_acc))env LANG=C.UTF-8

                # if dev_loss<init_dev_loss:
                #     init_dev_loss=dev_loss
                #     init_dev_acc=dev_intent_acc
                #     self.intent_write(dev_softmax_logit, dev_intent, dev_sent, dev_slot, id2sent, id2intent,
                #                       id2slot, 'dev_out')
                #     self.intent_write(train_softmax_logit, train_intent, train_sent, train_slot, id2sent, id2intent,
                #                       id2slot, 'train_out')
                #     saver.save(sess,'%s'%FLAGS.model_dir)
                #     _logger.info('save model')
                #
                # endtime=time.time()
                # print('time:%s'%(endtime-start_time))
                # _logger.info('\n')







if __name__ == '__main__':                

    with tf.device('/gpu:1'):
        data_transfer = data_transfer('pos.txt', 'neg.txt')
        data_transfer.words_dictionary()
        train_data = data_transfer.words_to_id(30)
        train_data, dev_data = data_transfer.train_dev_split(train_data, 0.8)
        nn_model = Graph( vocab_num=len(data_transfer.dictionary))
        nn_model.train(train_data, dev_data)



