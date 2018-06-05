'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import sys
sys.path.append('../')
import tensorflow as tf
from data_preprocess import Intent_Slot_Data
from trans_model.modules import *
import os, codecs
import time

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
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


config=Config

class Graph():
    def __init__(self,intent_num_class,vocab_num):
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        self.x = tf.placeholder(tf.int32, shape=(None, config.maxlen))
        self.intent_num_class=intent_num_class

        self.intent = tf.placeholder(shape=(None, self.intent_num_class), dtype=tf.int32)
        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                  vocab_size=vocab_num,
                                  num_units=config.hidden_units,
                                  scale=True,
                                  scope="enc_embed")

            # ## Positional Encoding
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
            #
            #
            # ## Dropout
            # self.enc = tf.layers.dropout(self.enc,
            #                             rate=config.dropout_rate,
            #                             training=True)
            #
            ## Blocks
            # for i in range(1):
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
            #
            # encoder=tf.nn.max_pool(enc, ksize = [1,config.maxlen, 1, 1], strides = [1, config.maxlen, 1, 1], padding = 'VALID', name = 'maxpool1')
            #
            encoder=tf.reduce_mean(self.enc,1)
            print(encoder)

            # filter_w_1 = tf.Variable(tf.truncated_normal(shape=(3, config.hidden_units, 1, 512), stddev=0.1), name="W")
            # filter_b_1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b")
            #
            # cnn_out1 = tf.nn.conv2d(enc, filter_w_1, strides=[1, 1, 1, 1], padding='VALID')  #
            # cnn_out1 = tf.nn.relu(tf.nn.bias_add(cnn_out1, filter_b_1))
            # cnn_out1 = tf.nn.max_pool(cnn_out1, ksize=[1, config.maxlen-3 + 1, 1, 1], strides=[1, 1, 1, 1],
            #                           padding='VALID')

            # encoder=tf.reshape(encoder,[-1,512])
            # encoder=tf.layers.dense(encoder,512,activation=tf.nn.relu,use_bias=True)

            # encoder=tf.transpose(self.enc,[1,0,2])[0]

            # print(encoder)
            # encoder=tf.reshape(self.enc,[-1,3000])
            self.sent_sim_emb=encoder
            lstm_w = tf.Variable(tf.random_normal(shape=(512, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))
            logit = tf.add(tf.matmul(encoder, lstm_w), lstm_b)
            self.soft_logit = tf.nn.softmax(logit, 1)
            l2_loss = tf.get_collection('l2')
            intent=tf.cast(self.intent,tf.float32)
            intent=label_smoothing(intent)
            intent_loss=-tf.reduce_sum(intent*tf.log(tf.clip_by_value(self.soft_logit,1e-5,1.0)))

            # intent_loss = tf.losses.softmax_cross_entropy(self.intent, logit , reduction=tf.losses.Reduction.NONE)
            # intent_loss = intent_loss
            # intent_loss = tf.reduce_mean(intent_loss)

            self.loss_op = intent_loss
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.4).minimize(self.loss_op)
            #
            # # 梯度截断
            self.opt = tf.train.AdamOptimizer(learning_rate=0.3)
            grads_vars = self.opt.compute_gradients(self.loss_op)
            capped_grads_vars = [[tf.clip_by_value(g, -1e-5, 1.0), v]
                                 for g, v in grads_vars]
            self.optimizer = self.opt.apply_gradients(capped_grads_vars)

            # # Decoder
            # with tf.variable_scope("decoder"):
            #     ## Embedding
            #     self.dec = embedding(self.decoder_inputs,
            #                           vocab_size=len(en2idx),
            #                           num_units=hp.hidden_units,
            #                           scale=True,
            #                           scope="dec_embed")
            #
            #     ## Positional Encoding
            #     if hp.sinusoid:
            #         self.dec += positional_encoding(self.decoder_inputs,
            #                           vocab_size=hp.maxlen,
            #                           num_units=hp.hidden_units,
            #                           zero_pad=False,
            #                           scale=False,
            #                           scope="dec_pe")
            #     else:
            #         self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
            #                           vocab_size=hp.maxlen,
            #                           num_units=hp.hidden_units,
            #                           zero_pad=False,
            #                           scale=False,
            #                           scope="dec_pe")
            #
            #     ## Dropout
            #     self.dec = tf.layers.dropout(self.dec,
            #                                 rate=hp.dropout_rate,
            #                                 training=tf.convert_to_tensor(is_training))
            #
            #     ## Blocks
            #     for i in range(hp.num_blocks):
            #         with tf.variable_scope("num_blocks_{}".format(i)):
            #             ## Multihead Attention ( self-attention)
            #             self.dec = multihead_attention(queries=self.dec,
            #                                             keys=self.dec,
            #                                             num_units=hp.hidden_units,
            #                                             num_heads=hp.num_heads,
            #                                             dropout_rate=hp.dropout_rate,
            #                                             is_training=is_training,
            #                                             causality=True,
            #                                             scope="self_attention")
            #
            #             ## Multihead Attention ( vanilla attention)
            #             self.dec = multihead_attention(queries=self.dec,
            #                                             keys=self.enc,
            #                                             num_units=hp.hidden_units,
            #                                             num_heads=hp.num_heads,
            #                                             dropout_rate=hp.dropout_rate,
            #                                             is_training=is_training,
            #                                             causality=False,
            #                                             scope="vanilla_attention")
            #
            #             ## Feed Forward
            #             self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
            #
            # # Final linear projection
            # self.logits = tf.layers.dense(self.dec, len(en2idx))
            # self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            # self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            # self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            # tf.summary.scalar('acc', self.acc)
            #
            # if is_training:
            #     # Loss
            #     self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
            #     self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            #     self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
            #
            #     # Training Scheme
            #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #     self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            #     self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            #
            #     # Summary
            #     tf.summary.scalar('mean_loss', self.mean_loss)
            #     self.merged = tf.summary.merge_all()


    def train(self,dd):
        import pickle
        train_emb_dict={}
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )

        num_batch=dd.num_batch
        id2sent=dd.id2sent
        id2intent=dd.id2intent
        id2slot=dd.id2slot
        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            dev_sent,dev_slot,dev_intent,dev_rel_len=dd.get_dev()
            train_sent,train_slot,train_intent,train_rel_len=dd.get_train()
            for j in range(100):
                _logger.info('第%s次epoch'%j)
                start_time = time.time()
                for i in range(num_batch):
                    sent,slot,intent,rel_len,cur_len=dd.next_batch()

                    intent_loss,softmax_logit, _ = sess.run([self.loss_op,self.soft_logit, self.optimizer], feed_dict={self.x: sent,
                                                                                         self.intent: intent,
                                                                                         })





                dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.x: dev_sent,
                                                                 self.intent: dev_intent,
                                                                 })
                dev_intent_acc=self.intent_acc(dev_softmax_logit,dev_intent)


                sent_emb_,train_softmax_logit, train_loss = sess.run([self.sent_sim_emb,self.soft_logit, self.loss_op],
                                                       feed_dict={self.x: train_sent,
                                                                  self.intent: train_intent,
                                                                  })
                for sent, sent_emb_ele in zip(train_sent, sent_emb_):
                    ss = ''.join([id2sent[e] for e in sent if e != 0])
                    if ss not in train_emb_dict:
                        train_emb_dict[ss] = sent_emb_ele
                pickle.dump(train_emb_dict, open('./train_sent_emb.p', 'wb'))

                train_intent_acc = self.intent_acc(train_softmax_logit, train_intent)

                _logger.info('train_intent_loss:%s train_intent_acc:%s'%(train_loss,train_intent_acc))
                _logger.info('dev_intent_loss:%s dev_intent_acc:%s'%(dev_loss,dev_intent_acc))

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


    def intent_acc(self,pre,label):
        '''
        获取intent准确率
        :param pre:
        :param label:
        :return:
        '''
        all_sum=len(pre)
        predict=[]
        for ele in pre:
            ss=[]
            for e in ele:
                if float(e)>0.3:
                    ss.append(1)
                else:
                    ss.append(0)
            if sum([1 for e in ss if e==0])==len(ss):
                max_index=np.argmax(ele)
                ss[max_index]=1
            predict.append(ss)

        num=0
        for predict_ele,label_ele in zip(predict,label):
            pre_=" ".join([str(index) for index,e in enumerate(predict_ele) if e ==1])
            label_=" ".join([str(index) for index,e in enumerate(label_ele) if e ==1])
            if pre_==label_:
                num+=1

        return float(num)/float(all_sum)






if __name__ == '__main__':                
    # Load vocabulary    
    # de2idx, idx2de = load_de_vocab()
    # en2idx, idx2en = load_en_vocab()
    #
    # # Construct graph
    # g = Graph("train"); print("Graph loaded")
    #
    # # Start session
    # sv = tf.train.Supervisor(graph=g.graph,
    #                          logdir=hp.logdir,
    #                          save_model_secs=0)
    # with sv.managed_session() as sess:
    #     for epoch in range(1, hp.num_epochs+1):
    #         if sv.should_stop(): break
    #         for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
    #             sess.run(g.train_op)
    #
    #         gs = sess.run(g.global_step)
    #         sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    #
    # print("Done")
    with tf.device('/gpu:1'):
        dd = Intent_Slot_Data(train_path="../dataset/train_out_char.txt",
                              test_path="../dataset/dev_out_char.txt",
                              dev_path="../dataset/dev_out_char.txt", batch_size=config.batch_size,
                              max_length=config.maxlen, flag="train_new",
                              use_auto_bucket=False)

        nn_model = Graph( intent_num_class=dd.intent_num, vocab_num=dd.vocab_num)
        nn_model.train(dd)


