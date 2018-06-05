import numpy as np
import sys
import os
from xmlrpc.server import SimpleXMLRPCServer
import tensorflow as tf
import logging
from pre_data_deal.data_deal import Intent_Data_Deal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
import time
from data_preprocess import Intent_Slot_Data
import xlwt
from configparser import ConfigParser
import gc
from focal_loss import focal_loss,focal_loss_1
Config=ConfigParser()
Config.read('Config.conf')
HOST=Config['host']['host']
LSTM_PORT=int(Config['lstm']['port'])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append("./")
sys.path.append("./data/")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")


class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 16
    sent_len = 30  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 100
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = './save_model/model_lstm1/intent_lstm.ckpt'
    if not os.path.exists('./save_model/model_lstm1'):
        os.makedirs('./save_model/model_lstm1')
    use_cpu_num = 16
    keep_dropout = 0.9
    summary_write_dir = "./tmp/r_net.log"
    epoch = 100
    use_auto_buckets=False
    lambda1 = 0.01
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf


config = Config_lstm()
tf.app.flags.DEFINE_float("lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("max_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "epoch次数")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_boolean('use Encoder2Decoder',False,'')
tf.app.flags.DEFINE_string("mod", "train", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_string('model_mode', config.model_mode, '模型类型')
tf.app.flags.DEFINE_boolean('use_auto_buckets',config.use_auto_buckets,'是否使用自动桶')
tf.app.flags.DEFINE_string('only_mode','intent','执行哪种单一任务')
FLAGS = tf.app.flags.FLAGS


class Model_LSTM(object):

    def __init__(self, slot_num_class,intent_num_class,vocab_num,intent_weight):


        self.hidden_dim = FLAGS.hidden_dim
        self.use_buckets=FLAGS.use_auto_buckets
        self.model_mode = FLAGS.model_mode
        self.batch_size = FLAGS.batch_size
        self.max_len=FLAGS.max_len
        self.embedding_dim = FLAGS.embedding_dim
        self.slot_num_class=slot_num_class
        self.intent_num_class=intent_num_class
        self.vocab_num=vocab_num
        self.intent_weight=intent_weight

        with tf.device('/gpu:2'):
            self.init_graph()
            self.encoder_outs,self.encoder_final_states=self.encoder()
            self.intent_losses=self.intent_loss()
            self.loss_op=self.intent_losses
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss_op)


    def init_graph(self):
        '''

        :return:
        '''
        if self.use_buckets:
            self.sent=tf.placeholder(shape=(None,None),dtype=tf.int32)
            self.slot=tf.placeholder(shape=(None,None),dtype=tf.int32)
            self.intent=tf.placeholder(shape=(None,None),dtype=tf.int32)
            self.seq_vec=tf.placeholder(shape=(None,),dtype=tf.int32)
            self.rel_num=tf.placeholder(shape=(1,),dtype=tf.int32)
        else:
            self.sent = tf.placeholder(shape=(None, self.max_len), dtype=tf.int32)
            self.slot = tf.placeholder(shape=(None, self.max_len), dtype=tf.int32)
            self.intent = tf.placeholder(shape=(None,self.intent_num_class), dtype=tf.int32)
            self.loss_weight=tf.placeholder(shape=(None,),dtype=tf.float32)
            self.seq_vec = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.rel_num = tf.placeholder(shape=(1,), dtype=tf.int32)

        # self.global_step = tf.Variable(0, trainable=True)

        self.length_embedding=tf.Variable(tf.random_normal(shape=(self.max_len+1,50)),trainable=False)

        self.sent_embedding=tf.Variable(tf.random_normal(shape=(self.vocab_num,self.embedding_dim),
                                                         dtype=tf.float32),trainable=True)
        self.slot_embedding=tf.Variable(tf.random_normal(shape=(self.slot_num_class,self.embedding_dim),
                                                         dtype=tf.float32),trainable=False)

        self.sent_emb=tf.nn.embedding_lookup(self.sent_embedding,self.sent)
        self.slot_emb=tf.nn.embedding_lookup(self.slot_embedding,self.slot)
        self.len_emb=tf.nn.embedding_lookup(self.length_embedding,self.seq_vec)
        if FLAGS.mod=='train':
            self.sent_emb=tf.nn.dropout(self.sent_emb,FLAGS.keep_dropout)

        self.lstm_fw=tf.contrib.rnn.LSTMCell(self.hidden_dim)
        self.lstm_bw=tf.contrib.rnn.LSTMCell(self.hidden_dim)
        if FLAGS.mod=='train':
            self.lstm_fw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_fw, output_keep_prob=FLAGS.keep_dropout)
            self.lstm_bw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_bw, output_keep_prob=FLAGS.keep_dropout)



    def encoder(self):
        '''
        编码层
        :return:
        '''
        #final_states=((fw_c_last,fw_h_last),(bw_c_last,bw_h_last))
        lstm_out, final_states = tf.nn.bidirectional_dynamic_rnn(
            self.lstm_fw,
            self.lstm_bw,
            self.sent_emb,
            dtype=tf.float32,
            sequence_length=self.seq_vec,)

        lstm_out=tf.concat(lstm_out,2)
        lstm_outs=tf.stack(lstm_out) # [batch_size,seq_len,dim] 作为attention的注意力矩阵

        state_c=tf.concat((final_states[0][0],final_states[1][0]),1) #作为decoder的inital states中state_c

        state_h=tf.concat((final_states[0][1],final_states[1][1]),1) #作为decoder的inital states中state_h

        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            c=state_c,
            h=state_h
        )
        return lstm_outs,encoder_final_state

    def cnn_encoder(self):
        '''
        cnn编码
        :return:
        '''
        input_emb=tf.expand_dims(self.sent_emb,3)
        filter=[4,6,9,12]
        res=[]
        for index,ele in enumerate(filter):
            with tf.name_scope("conv-maxpool-%s" % index):
                filter_w_1 = tf.Variable(tf.truncated_normal(shape=(ele,self.embedding_dim,1,200), stddev=0.1), name="W")
                filter_b_1 = tf.Variable(tf.constant(0.1, shape=[200]), name="b")


                cnn_out1=tf.nn.conv2d(input_emb,filter_w_1,strides=[1,1,1,1],padding='VALID') #
                cnn_out1=tf.nn.relu(tf.nn.bias_add(cnn_out1,filter_b_1))
                cnn_out1=tf.nn.max_pool(cnn_out1,ksize=[1,self.max_len-ele+1,1,1],strides=[1,2,1,1],padding='VALID')


                res.append(cnn_out1)
        ress=tf.concat(res,3)
        cnn_out=tf.reshape(ress,[-1,800])
        cnn_out=tf.nn.dropout(cnn_out,0.7)
        # sent_attention=self.intent_attention(self.sent_emb)
        # cnn_out=tf.concat((cnn_out,sent_attention),1)
        return cnn_out


    def intent_attention(self, lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, 2 * self.hidden_dim)))
        b_h = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim,)))
        logit = tf.einsum("ijk,kl->ijl", lstm_outs, w_h)
        logit = tf.nn.tanh(tf.add(logit, b_h))
        logit = tf.tanh(tf.einsum("ijk,ilk->ijl", logit, lstm_outs))
        G = tf.nn.softmax(logit)  # G.shape=[self.seq_len,self.seq_len]
        logit_ = tf.einsum("ijk,ikl->ijl", G, lstm_outs)

        # 注意力得到的logit与lstm_outs进行链接

        outs = tf.concat((logit_, lstm_outs), 2)  # outs.shape=[None,seq_len,4*hidden_dim]
        return outs

    def intent_attention_1(self, lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, 2*self.hidden_dim)))
        b_h = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,)))
        v_h = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,1)))
        logit = tf.einsum("ijk,kl->ijl", lstm_outs, w_h)
        logit = tf.nn.tanh(tf.add(logit, b_h))
        logit =tf.einsum('ijk,kl->ijl',logit,v_h)
        logit=tf.reshape(logit,shape=(-1,self.max_len))
        G = tf.nn.softmax(logit,1)  # G.shape=[self.seq_len,self.seq_len]
        # logit = tf.tanh(tf.einsum("ijk,ilk->ijl", logit, lstm_outs))
        logit_ = tf.einsum("ikj,ik->ij", lstm_outs,G)
        logit_=tf.reshape(logit_,[-1,2*self.hidden_dim])
        s=tf.transpose(lstm_outs,[1,0,2])[0]
        outs = tf.concat((logit_, s), 1)
        return outs

    def self_lstm_attention_ops(self,lstm_out_t,lstm_outs):
        '''

        :return:
        '''
        w=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_out_t 参数
        g=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_outs 参数
        lstm_out_t=tf.reshape(lstm_out_t,[-1,1,2*self.hidden_dim])

        v=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,1)))
        with tf.variable_scope('self_attention',reuse=True):
            lstm_out_t_=tf.einsum('ijk,kl->ijl',lstm_out_t,w)
            lstm_outs_=tf.einsum('ijk,kl->ijl',lstm_outs,g)
            gg=tf.tanh(lstm_out_t_+lstm_outs_)
            gg_=tf.einsum('ijk,kl->ijl',gg,v)
            gg_soft=tf.nn.softmax(gg_,1)
            a=tf.einsum('ijk,ijl->ikl',lstm_outs,gg_soft)
            a=tf.reshape(a,[-1,2*self.hidden_dim])
            return a

    def attention(self,lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,2*self.hidden_dim)))
        b_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        logit=tf.nn.tanh(tf.add(logit,b_h))
        logit=tf.tanh(tf.einsum("ijk,ilk->ijl",logit,lstm_outs))
        G=tf.nn.softmax(logit)#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)


        # 注意力得到的logit与lstm_outs进行链接

        outs=tf.concat((logit_,lstm_outs),2)#outs.shape=[None,seq_len,4*hidden_dim]
        return outs

    def self_lstm_attention_ops_decoder(self,lstm_out_t,lstm_outs):
        '''

        :return:
        '''
        w=tf.Variable(tf.random_uniform(shape=(self.hidden_dim,2*self.hidden_dim))) #lstm_out_t 参数
        g=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_outs 参数
        lstm_out_t=tf.reshape(lstm_out_t,[-1,1,self.hidden_dim])

        out_w_=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,self.hidden_dim)))
        out_b_=tf.Variable(tf.random_uniform(shape=(self.hidden_dim,)))

        v=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,1)))
        with tf.variable_scope('self_attention',reuse=True):
            lstm_out_t_=tf.einsum('ijk,kl->ijl',lstm_out_t,w)
            lstm_outs_=tf.einsum('ijk,kl->ijl',lstm_outs,g)
            gg=tf.tanh(lstm_out_t_+lstm_outs_)
            gg_=tf.einsum('ijk,kl->ijl',gg,v)
            gg_soft=tf.nn.softmax(gg_,1)
            a=tf.einsum('ijk,ijl->ikl',lstm_outs,gg_soft)
            a=tf.reshape(a,[-1,2*self.hidden_dim])
            new_a=tf.add(tf.matmul(a,out_w_),out_b_)
            return new_a

    def self_lstm_attention(self,lstm_outs):
        '''
        对lstm输出再做一层 attention_lstm
        :param lstm_outs:
        :return:
        '''

        lstm_cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                state_is_tuple=True)
        lstm_outs_list=tf.unstack(lstm_outs,self.max_len,1)
        init_state=tf.zeros_like(lstm_outs_list[0])
        states=[(init_state,init_state)]
        H=[]
        w=tf.Variable(tf.random_uniform(shape=(4*self.hidden_dim,4*self.hidden_dim)))
        with tf.variable_scope('lstm_attention'):
            for i in range(self.max_len):
                if i>0:
                    tf.get_variable_scope().reuse_variables()
                lstm_outs_t=lstm_outs_list[i]
                a=self.self_lstm_attention_ops(lstm_outs_t,lstm_outs) #attention的值

                new_input=tf.concat((lstm_outs_t,a),1)

                new_input_=tf.sigmoid(tf.matmul(new_input,w))*new_input

                h,state=lstm_cell(new_input_,states[-1])
                H.append(h)
                states.append(state)
        H=tf.stack(H)
        H=tf.transpose(H,[1,0,2])
        return H

    def intent_loss(self):
        '''

        :return:
        '''

        intent_mod='origin_attention'

        if intent_mod=='max_pool':
            lstm_w = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))

            encoder_out=tf.expand_dims(self.encoder_outs,3)
            lstm_out=tf.nn.max_pool(encoder_out, ksize = [1,self.rel_num, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'maxpool1')
            lstm_out=tf.reshape(lstm_out,[-1,2*self.hidden_dim])
            logit=tf.add(tf.matmul(lstm_out,lstm_w),lstm_b)
            intent_one_hot=tf.one_hot(self.intent,self.intent_num_class,1,0)
            intent_loss=tf.losses.softmax_cross_entropy(intent_one_hot,logit)
            return intent_loss

        elif intent_mod=='origin_attention':
            lstm_w = tf.Variable(tf.random_normal(shape=(4 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            tf.add_to_collection('l2',tf.contrib.layers.l2_regularizer(FLAGS.lambda1)(lstm_w))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))
            lstm_out=self.intent_attention(self.encoder_outs)
            lstm_out=tf.transpose(lstm_out,[1,0,2])[0]
            logit = tf.add(tf.matmul(lstm_out, lstm_w), lstm_b)
            self.soft_logit=tf.nn.softmax(logit,1)
            l2_loss=tf.get_collection('l2')
            # intent_loss=focal_loss_1(self.soft_logit,tf.cast(self.intent,tf.float32),weights=self.intent_weight)
            intent_loss=focal_loss(self.soft_logit,tf.cast(self.intent,tf.float32))

            # intent_loss=tf.losses.softmax_cross_entropy(self.intent,logit,reduction=tf.losses.Reduction.NONE)
            # intent_loss=intent_loss
            # intent_loss=tf.reduce_mean(intent_loss)
            return intent_loss+l2_loss

        elif intent_mod=='origin_self_attenion':
            lstm_w = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))
            lstm_out=self.self_lstm_attention(self.encoder_outs)
            lstm_out=tf.transpose(lstm_out,[1,0,2])[-1]
            logit = tf.add(tf.matmul(lstm_out, lstm_w), lstm_b)
            self.soft_logit=tf.nn.softmax(logit,1)
            intent_one_hot = tf.one_hot(self.intent, self.intent_num_class, 1, 0)
            intent_loss=tf.losses.softmax_cross_entropy(self.intent,logit,reduction=tf.losses.Reduction.NONE)
            # mask=tf.sequence_mask(self.seq_vec,self.intent_num_class)
            # intent_loss=tf.boolean_mask(loss,mask)
            intent_loss=tf.reduce_mean(intent_loss)
            return intent_loss

    def train(self,dd):


        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        saver=tf.train.Saver()
        init_dev_loss=99999.99
        init_train_loss=999.99
        init_dev_acc=0.0
        num_batch=dd.num_batch
        id2sent=dd.id2sent
        id2intent=dd.id2intent
        id2slot=dd.id2slot

        with tf.Session(config=config) as sess:
            if os.path.exists('%s.meta'%FLAGS.model_dir):
                saver.restore(sess,'%s'%FLAGS.model_dir)
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent,dev_slot,dev_intent,dev_rel_len=dd.get_dev()
            train_sent,train_slot,train_intent,train_rel_len=dd.get_train()
            for j in range(FLAGS.epoch):
                _logger.info('第%s次epoch'%j)
                start_time = time.time()
                for i in range(num_batch):
                    sent,slot,intent,rel_len,cur_len=dd.next_batch()

                    intent_loss,softmax_logit, _ = sess.run([self.loss_op,self.soft_logit, self.optimizer], feed_dict={self.sent: sent,
                                                                                         self.slot: slot,
                                                                                         self.intent: intent,
                                                                                         self.seq_vec: rel_len,
                                                                                         self.rel_num: cur_len,
                                                                                         })





                dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.sent: dev_sent,
                                                                 self.slot: dev_slot,
                                                                 self.intent: dev_intent,
                                                                 self.seq_vec: dev_rel_len,
                                                                 })
                dev_intent_acc=self.intent_acc(dev_softmax_logit,dev_intent)


                train_softmax_logit, train_loss = sess.run([self.soft_logit, self.loss_op],
                                                       feed_dict={self.sent: train_sent,
                                                                  self.slot: train_slot,
                                                                  self.intent: train_intent,
                                                                  self.seq_vec: train_rel_len,
                                                                  })
                train_intent_acc = self.intent_acc(train_softmax_logit, train_intent)

                _logger.info('train_intent_loss:%s train_intent_acc:%s'%(train_loss,train_intent_acc))
                _logger.info('dev_intent_loss:%s dev_intent_acc:%s'%(dev_loss,dev_intent_acc))

                if dev_loss<init_dev_loss:
                    init_dev_loss=dev_loss
                    init_dev_acc=dev_intent_acc
                    # self.intent_write(dev_softmax_logit, dev_intent, dev_sent, dev_slot, id2sent, id2intent,
                    #                   id2slot, 'dev_out')
                    # self.intent_write(train_softmax_logit, train_intent, train_sent, train_slot, id2sent, id2intent,
                    #                   id2slot, 'train_out')
                    saver.save(sess,'%s'%FLAGS.model_dir)
                    _logger.info('save model')

                endtime=time.time()
                print('time:%s'%(endtime-start_time))
                _logger.info('\n')


    def matirx(self,pre_label,label,id2intent,file_name):

        pre_label=np.argmax(pre_label,1)
        label=np.argmax(label,1)

        pre_label=pre_label.flatten()
        label=label.flatten()

        labels = list(set(label))


        target_names=[id2intent[e] for e in labels]


        conf_mat = confusion_matrix(label, pre_label, labels=labels)

        # rb1 = xlwt.Workbook()
        # sheet = rb1.add_sheet(u'sheet1', cell_overwrite_ok=True)
        #
        # for i in range(1,len(labels)+1):
        #     sheet.write(i,0,id2intent[int(labels[i-1])])
        #     sheet.write(0,i,id2intent[int(labels[i-1])])
        # for i in range(1,conf_mat.shape[0]+1):
        #     ii=1
        #     for j in range(1,conf_mat.shape[1]+1):
        #         if conf_mat[i-1,j-1]!=0:
        #             sheet.write(i,ii,id2intent[labels[j-1]]+str(conf_mat[i-1,j-1]))
        #             ii+=1
        # rb1.save('confusion_%s' % file_name)


        # print(conf_mat)
        class_re=classification_report(label,pre_label,target_names=target_names)
        print(class_re)
        # fw=open(file_name,'w')
        # fw.write(class_re)
        # print(class_re)
        # p, r, f1, s = precision_recall_fscore_support(label, pre_label,
        #                                               labels=labels,
        #                                               average=None,
        #                                               sample_weight=None)
        #
        # rb = xlwt.Workbook()
        # sheet = rb.add_sheet(u'sheet1', cell_overwrite_ok=True)
        # index=1
        # for name,p_,f_,f1_,s_ in zip(target_names,p,r,f1,s):
        #     sheet.write(index, 1, name)
        #     sheet.write(index, 2, p_)
        #     sheet.write(index, 3, f_)
        #     sheet.write(index, 4, f1_)
        #     sheet.write(index, 5, int(s_))
        #
        #     index+=1
        # rb.save('%s'%file_name)


        # for name,p_,f_,f1_,s_ in zip(target_names,p,r,f1,s):
        #     print(name,p_,f_,f1_,s_)


    def infer_dev(self,dd):

        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        id2sent=dd.id2sent
        id2intent=dd.id2intent
        saver=tf.train.Saver()
        with tf.Session(config=config) as sess:
            if os.path.exists('%s.meta'%FLAGS.model_dir):
                saver.restore(sess,'%s'%FLAGS.model_dir)
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent,dev_slot,dev_intent,dev_rel_len=dd.get_dev()
            train_sent,train_slot,train_intent,train_rel_len=dd.get_train()


            dev_softmax_logit, dev_loss = sess.run([self.soft_logit, self.loss_op], feed_dict={self.sent: dev_sent,
                                                                                               self.slot: dev_slot,
                                                                                               self.intent: dev_intent,
                                                                                               self.seq_vec: dev_rel_len,
                                                                                               })

            self.matirx(dev_softmax_logit,dev_intent,id2intent,'dev.xlsx')
            self.intent_write(dev_softmax_logit,dev_intent,dev_sent,id2sent,id2intent,'dev_out')
            print('\n\n')

            train_softmax_logit, train_loss = sess.run([self.soft_logit, self.loss_op],
                                                       feed_dict={self.sent: train_sent,
                                                                  self.slot: train_slot,
                                                                  self.intent: train_intent,
                                                                  self.seq_vec: train_rel_len,
                                                                  })
            self.matirx(train_softmax_logit,train_intent,id2intent,'train.xlsx')
            self.intent_write(train_softmax_logit,train_intent,train_sent,id2sent,id2intent,'train_out')


    def infer(self,dd,sent,sess):
        '''

        :param dd:
        :param sent:
        :return:
        '''

        id2intent=dd.id2intent

        sent_arr,sent_vec=dd.get_sent_char(sent)


        intent_logit=sess.run(self.soft_logit,feed_dict={self.sent:sent_arr,
                                            self.seq_vec:sent_vec})

        res=[]
        for ele in intent_logit:
            ss=[[id2intent[index],str(e)] for index,e in enumerate(ele) if e>=0.3]
            if not ss:
                ss=[[id2intent[np.argmax(ele)],str(np.max(ele))]]
            res.append(ss)
        return res

    def intent_write(self,pre,label,sent,id2sent,id2intent,file_name):
        '''
        写入txt
        :param pre:
        :param label:
        :param sent:
        :return:
        '''
        fw1=open('./%s_对比.txt'%file_name,'w')
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

        for predict_ele, label_ele,sent_ele in zip(predict, label,sent):
            pre_ = " ".join([id2intent[index] for index, e in enumerate(predict_ele) if e == 1])
            label_ = " ".join([id2intent[index] for index, e in enumerate(label_ele) if e == 1])

            sent=' '.join([e for e in [id2sent[e] for e in sent_ele] if e!='NONE'])
            if pre_!=label_:
                fw1.write(pre_)
                fw1.write('\t\t')
                fw1.write(label_)
                fw1.write('\t\t')
                fw1.write(sent)
                fw1.write('\n')

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

def build_intent_weight(intent_vocab,intent_num_vocab):

    intent_vocab = intent_vocab
    all_num = 0.0
    ss = [v for k, v in intent_num_vocab.items()]
    ss.sort(reverse=True)
    max_length = ss[0]
    intent_rate_list = []
    for k, v in intent_num_vocab.items():
        rate = 1.1 - (float(v) / float(max_length))
        intent_rate_list.append([intent_vocab[k], rate])
    # print(dd.intent_num_vocab)

    intent_rate_list.sort(key=lambda x: x[0])
    intent_weight = np.array([e[1] for e in intent_rate_list],dtype=np.float32)
    return intent_weight


def main(_):
    start_time = time.time()
    config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            )
    with tf.device("/gpu:2"):
        _logger.info("load data")
        dd = Intent_Slot_Data(train_path="./dataset/train_out_char.txt",
                              test_path="./dataset/dev_out_char.txt",
                              dev_path="./dataset/dev_out_char.txt", batch_size=FLAGS.batch_size,
                              max_length=FLAGS.max_len, flag="train_new",
                              use_auto_bucket=FLAGS.use_auto_buckets)

        intent_vocab=dd.intent_vocab
        intent_num_vocab=dd.intent_num_vocab
        intent_weight=build_intent_weight(intent_vocab,intent_num_vocab)
        nn_model = Model_LSTM(slot_num_class=dd.slot_num, intent_num_class=dd.intent_num, vocab_num=dd.vocab_num,intent_weight=intent_weight)
        if FLAGS.mod == 'train':
            nn_model.train(dd)

        elif FLAGS.mod == 'infer':
            idd = Intent_Data_Deal()
            nn_model.infer_dev(dd)

        elif FLAGS.mod=='server':

            saver = tf.train.Saver()
            sess = tf.Session(config=config)
            if os.path.exists('%s.meta' % FLAGS.model_dir):
                saver.restore(sess, '%s' % FLAGS.model_dir)
            else:
                _logger.error('lstm没有模型')

            idd = Intent_Data_Deal()
            def intent(sent_list):
                sents=[]
                _logger.info("%s"%len(sent_list))
                for sent in sent_list:
                    # sent=idd.deal_sent(sent)
                    sents.append(sent)
                all_res = nn_model.infer(dd, sents,sess)
                del sents
                gc.collect()
                _logger.info('process end')
                return all_res

            svr = SimpleXMLRPCServer((HOST, LSTM_PORT), allow_none=True)
            svr.register_function(intent)
            svr.serve_forever()

if __name__ == '__main__':

   tf.app.run()