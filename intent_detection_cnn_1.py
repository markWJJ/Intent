import numpy as np
import sys
import os
from xmlrpc.server import SimpleXMLRPCServer
import random
import tensorflow as tf
import logging
from logging.config import fileConfig
from pre_data_deal.data_deal import Intent_Data_Deal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
import time
from focal_loss import focal_loss
import pickle
from data_preprocess import Intent_Slot_Data
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("./")
sys.path.append("./data/")
from configparser import ConfigParser

Config=ConfigParser()
Config.read('Config.conf')
HOST=Config['host']['host']
# CNN_PORT=int(Config['cnn']['port'])

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")


class Config(object):
    '''
    默认配置
    '''
    learning_rate = 0.4
    batch_size = 16
    sent_len = 30  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 100
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = './save_model/model_cnn/intent_cnn.ckpt'
    if not os.path.exists('./save_model/model_cnn'):
        os.makedirs('./save_model/model_cnn')
    use_cpu_num = 16
    keep_dropout = 0.7
    summary_write_dir = "./tmp/cnn.log"
    epoch = 200
    use_auto_buckets=False
    lambda1 = 0.01
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf


config = Config()
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


class Model(object):

    def __init__(self,intent_num_class,vocab_num):

        with tf.device('/gpu:1'):
            self.hidden_dim = FLAGS.hidden_dim
            self.use_buckets=FLAGS.use_auto_buckets
            self.model_mode = FLAGS.model_mode
            self.batch_size = FLAGS.batch_size
            self.max_len=FLAGS.max_len
            self.embedding_dim = FLAGS.embedding_dim
            self.intent_num_class=intent_num_class
            self.vocab_num=vocab_num
            self.init_graph()


            self.cnn_out=self.cnn_encoder()


            self.intent_losses=self.intent_loss()
            self.loss_op=self.intent_losses
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss_op)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
            grads_vars = self.opt.compute_gradients(self.loss_op)
            # capped_grads_vars = [[tf.clip_by_value(g, -1e-5, 10.0), v]
            #                      for g, v in grads_vars]
            self.optimizer = self.opt.apply_gradients(grads_vars)

    def init_graph(self):
        '''

        :return:
        '''

        self.sent = tf.placeholder(shape=(None, self.max_len), dtype=tf.int32)
        self.intent = tf.placeholder(shape=(None,self.intent_num_class), dtype=tf.int32)
        self.loss_weight=tf.placeholder(shape=(None,),dtype=tf.float32)
        self.seq_vec = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.rel_num = tf.placeholder(shape=(1,), dtype=tf.int32)

        self.intent_embedding=tf.Variable(tf.random_normal(shape=(self.intent_num_class,300),dtype=tf.float32),trainable=True)

        # self.global_step = tf.Variable(0, trainable=True)

        self.length_embedding=tf.Variable(tf.random_normal(shape=(self.max_len+1,50)),trainable=False)

        self.sent_embedding=tf.Variable(tf.random_normal(shape=(self.vocab_num,self.embedding_dim),
                                                         dtype=tf.float32),trainable=True)

        self.sent_emb=tf.nn.embedding_lookup(self.sent_embedding,self.sent)
        self.len_emb=tf.nn.embedding_lookup(self.length_embedding,self.seq_vec)

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
        w_h = tf.Variable(tf.random_normal(shape=(self.embedding_dim, 2*self.embedding_dim)))
        b_h = tf.Variable(tf.random_normal(shape=(2*self.embedding_dim,)))
        v_h = tf.Variable(tf.random_normal(shape=(2*self.embedding_dim,1)))
        logit = tf.einsum("ijk,kl->ijl", lstm_outs, w_h)
        logit = tf.nn.tanh(tf.add(logit, b_h))
        logit =tf.einsum('ijk,kl->ijl',logit,v_h)
        logit=tf.reshape(logit,shape=(-1,self.max_len))
        G = tf.nn.softmax(logit,1)  # G.shape=[self.seq_len,self.seq_len]
        # logit = tf.tanh(tf.einsum("ijk,ilk->ijl", logit, lstm_outs))
        logit_ = tf.einsum("ikj,ik->ij", lstm_outs,G)
        return logit_

    def cnn_encoder(self):
        '''
        cnn编码
        :return:
        '''
        input_emb=tf.expand_dims(self.sent_emb,3)
        filter=[3,4,5,6]
        res=[]
        for index,ele in enumerate(filter):
            with tf.name_scope("conv-maxpool-%s" % index):
                filter_w_1 = tf.Variable(tf.truncated_normal(shape=(ele,self.embedding_dim,1,200), stddev=0.1), name="W")
                filter_b_1 = tf.Variable(tf.constant(0.1, shape=[200]), name="b")


                cnn_out1=tf.nn.conv2d(input_emb,filter_w_1,strides=[1,1,1,1],padding='VALID') #
                cnn_out1=tf.nn.relu(tf.nn.bias_add(cnn_out1,filter_b_1))
                cnn_out1=tf.nn.max_pool(cnn_out1,ksize=[1,self.max_len-ele+1,1,1],strides=[1,1,1,1],padding='VALID')


                res.append(cnn_out1)
        ress=tf.concat(res,3)
        cnn_out=tf.reshape(ress,[-1,800])
        cnn_out=tf.nn.dropout(cnn_out,FLAGS.keep_dropout)
        sent_attention=self.intent_attention(self.sent_emb)
        cnn_out=tf.concat((cnn_out,sent_attention),1)
        return cnn_out

    def label_smoothing(self,inputs, epsilon=0.1):
        '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

        Args:
          inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
          epsilon: Smoothing rate.

        For example,

        ```
        import tensorflow as tf
        inputs = tf.convert_to_tensor([[[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]],

          [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]], tf.float32)

        outputs = label_smoothing(inputs)

        with tf.Session() as sess:
            print(sess.run([outputs]))

        >>
        [array([[[ 0.03333334,  0.03333334,  0.93333334],
            [ 0.03333334,  0.93333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334]],

           [[ 0.93333334,  0.03333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334],
            [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        ```
        '''
        K = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / K)

    def discriminator_layer(self,input,out_dim,reuse=False):

        with tf.variable_scope(name_or_scope='out_layer',reuse=reuse):
            w=tf.Variable(tf.random_uniform(shape=(300,out_dim),dtype=tf.float32))
            b=tf.Variable(tf.random_uniform(shape=(out_dim,),dtype=tf.float32))
            out=tf.add(tf.matmul(input,w),b)
            return out


    def intent_loss(self):
        '''

        :return:
        '''


        cnn_out=tf.layers.dense(self.cnn_out,300,activation=tf.nn.tanh)
        cnn_out=self.discriminator_layer(cnn_out,self.intent_num_class,False)
        intent_embedding=self.discriminator_layer(self.intent_embedding,self.intent_num_class,True)
        self.sent_sim_smb=cnn_out
        sent_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(cnn_out), axis=1))
        intent_norms=tf.unstack(intent_embedding,self.intent_num_class,0)
        cosins=[]
        # 内积
        for ele in intent_norms:
            intent_norm = tf.sqrt(tf.reduce_sum(tf.square(ele)))
            ele=tf.expand_dims(ele,-1)
            sent_intent = tf.matmul(cnn_out,ele)
            sent_intent=tf.reshape(sent_intent,[-1,])
            cosin=sent_intent/(sent_emb_norm*intent_norm)
            cosins.append(cosin)
        cosin=tf.stack(cosins,1)
        self.consin=cosin
        self.soft_logit=tf.nn.softmax(self.consin,1)
        intent=self.label_smoothing(tf.cast(self.intent,tf.float32))

        intent_loss=focal_loss(self.soft_logit,intent)

        class_y = tf.constant(name='class_y', shape=[self.intent_num_class, self.intent_num_class], dtype=tf.float32,
                              value=np.identity(self.intent_num_class), )

        label_loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=class_y,logits=intent_embedding))

        # intent_loss = -tf.reduce_sum(intent * tf.log(tf.clip_by_value(self.soft_logit, 1e-5, 1.0, name=None)))

        # intent_loss=tf.losses.softmax_cross_entropy(self.intent,cosin)

        # logit=tf.clip_by_value(self.soft_logit,1e-5,1.0)
        # intent_label=tf.cast(self.intent,tf.float32)
        # intent_loss=-tf.reduce_mean(intent_label*tf.log(logit))


        # cosin = x3_x4 / (x3_norm * x4_norm)
        # cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))

        # cnn_w = tf.Variable(tf.random_normal(shape=(900, self.intent_num_class), dtype=tf.float32))
        # cnn_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,),dtype=tf.float32))
        # tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(FLAGS.lambda1)(cnn_w))
        # logit = tf.nn.xw_plus_b(self.cnn_out, cnn_w, cnn_b)
        # self.soft_logit=tf.nn.softmax(logit,1)
        # l2_loss=tf.get_collection('l2')
        # intent=tf.cast(self.intent,tf.float32)
        # intent_loss = -tf.reduce_sum(intent * tf.log(self.soft_logit))

        # intent_loss=tf.losses.softmax_cross_entropy(self.intent,logit,reduction=tf.losses.Reduction.NONE)
        # intent_loss=intent_loss
        # intent_loss=tf.reduce_mean(intent_loss)
        # 梯度截断

        return intent_loss+0.5*label_loss


    def train(self,dd):

        train_emb_dict={}

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
                start_time = time.time()
                _logger.info('第%s次epoch'%j)
                for i in range(num_batch):
                    sent,slot,intent,rel_len,cur_len=dd.next_batch()

                    intent_loss,softmax_logit, _ = sess.run([self.loss_op,self.soft_logit, self.optimizer], feed_dict={self.sent: sent,
                                                                                         self.intent: intent,
                                                                                         self.seq_vec: rel_len,
                                                                                         self.rel_num: cur_len,
                                                                                         })




                dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.sent: dev_sent,
                                                                 self.intent: dev_intent,
                                                                 self.seq_vec: dev_rel_len,
                                                                 })
                dev_intent_acc=self.intent_acc(dev_softmax_logit,dev_intent)

                sent_emb_,train_softmax_logit, train_loss = sess.run([self.sent_sim_smb,self.soft_logit, self.loss_op],
                                                       feed_dict={self.sent: train_sent,
                                                                  self.intent: train_intent,
                                                                  self.seq_vec: train_rel_len,
                                                                  })
                for sent, sent_emb_ele in zip(train_sent, sent_emb_):
                    ss = ''.join([id2sent[e] for e in sent if e != 0])
                    if ss not in train_emb_dict:
                        train_emb_dict[ss] = sent_emb_ele
                pickle.dump(train_emb_dict, open('./train_sent_emb_cnn.p', 'wb'))

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


                endtime=time.time()
                print('time:%s'%(endtime-start_time))
                _logger.info('\n')


    def matirx(self,pre_label,label,id2intent,file_name):

        pre_label=np.argmax(pre_label,1)
        label=np.argmax(label,1)

        pre_label=pre_label.flatten()
        label=label.flatten()

        labels = list(set(label))
        conf_mat = confusion_matrix(label, pre_label, labels=labels)
        # print(conf_mat)
        target_names=[id2intent[e] for e in labels]
        class_re=classification_report(label,pre_label,target_names=target_names)
        fw=open(file_name,'w')
        fw.write(class_re)
        print(class_re)
        p, r, f1, s = precision_recall_fscore_support(label, pre_label,
                                                      labels=labels,
                                                      average=None,
                                                      sample_weight=None)

        # for name,p_,f_,f1_,s_ in zip(target_names,p,r,f1,s):
        #     print(name,p_,f_,f1_,s_)


    def infer_dev(self,dd):

        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
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

            self.matirx(dev_softmax_logit,dev_intent,id2intent,'dev_f1.txt')
            print('\n\n')

            train_softmax_logit, train_loss = sess.run([self.soft_logit, self.loss_op],
                                                       feed_dict={self.sent: train_sent,
                                                                  self.slot: train_slot,
                                                                  self.intent: train_intent,
                                                                  self.seq_vec: train_rel_len,
                                                                  })
            self.matirx(train_softmax_logit,train_intent,id2intent,'train_f1.txt')

    def infer(self,dd,sent):
        '''

        :param dd:
        :param sent:
        :return:
        '''
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        saver = tf.train.Saver()
        id2intent=dd.id2intent
        with tf.Session(config=config) as sess:
            saver.restore(sess,'%s'%FLAGS.model_dir)

            sent_arr,sent_vec=dd.get_sent_char(sent)


            intent_logit,intent_cosin=sess.run([self.soft_logit,self.consin],feed_dict={self.sent:sent_arr,
                                                self.seq_vec:sent_vec})

            res=[]
            for ele in intent_cosin:
                ss=[[id2intent[index],e] for index,e in enumerate(ele)]
                ss.sort(key=lambda x:x[1],reverse=True)
                res.append(ss[:5])
            return res

    def intent_write(self,pre,label,sent,slot,id2sent,id2intent,id2slot,file_name):
        '''
        写入txt
        :param pre:
        :param label:
        :param sent:
        :return:
        '''
        fw=open('./%s.txt'%file_name,'w')
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

        for predict_ele, label_ele,sent_ele,slot_ele in zip(predict, label,sent,slot):
            pre_ = " ".join([id2intent[index] for index, e in enumerate(predict_ele) if e == 1])
            label_ = " ".join([id2intent[index] for index, e in enumerate(label_ele) if e == 1])
            if pre_ != label_:
                loss_weight=1.0
            else:
                loss_weight=1.0

            sent=' '.join([e for e in [id2sent[e] for e in sent_ele] if e!='NONE'])
            slot=' '.join([e for e in [id2slot[e] for e in slot_ele] if e!='NONE'])
            fw.write(sent)
            fw.write('\t')
            fw.write(slot)
            fw.write('\t')
            fw.write(label_)
            fw.write('\t')
            fw.write(str(loss_weight))
            fw.write('\n')
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
        pre_ = np.argmax(pre, 1)

        label_ = np.argmax(label, 1)
        all_sum = len(label_)
        num = sum([1 for e, e1 in zip(pre_, label_) if e == e1])

        return float(num)/float(all_sum)



def main(_):
    with tf.device("/gpu:2"):
        _logger.info("load data")
        dd = Intent_Slot_Data(train_path="./dataset/train_out_char.txt",
                              test_path="./dataset/dev_out_char.txt",
                              dev_path="./dataset/dev_out_char.txt", batch_size=FLAGS.batch_size,
                              max_length=FLAGS.max_len, flag="train_new",
                              use_auto_bucket=FLAGS.use_auto_buckets)

        # sent, slot, intent, rel_len, cur_len = dd.next_batch()
        # _logger.info('input_param:{}'.format(sent.shape,slot.shape,intent.shape,rel_len.shape,cur_len.shape))
        nn_model = Model(intent_num_class=dd.intent_num, vocab_num=dd.vocab_num)
        if FLAGS.mod == 'train':
            nn_model.train(dd)

        elif FLAGS.mod == 'infer':
            idd = Intent_Data_Deal()
            # nn_model.infer_dev(dd)
            while True:
                sent = input('输入')
                sent = idd.deal_sent(sent)
                print(sent)
                res = nn_model.infer(dd, [sent])
                print(res)

        elif FLAGS.mod=='server':
            idd = Intent_Data_Deal()
            def intent(sent_list):
                sents=[]
                _logger.info("%s"%sent_list)
                for sent in sent_list:
                    # sent=idd.deal_sent(sent)
                    sents.append(sent)
                all_res = nn_model.infer(dd, sents)
                _logger.info('process end')
                return all_res

            svr = SimpleXMLRPCServer((HOST, CNN_PORT), allow_none=True)
            svr.register_function(intent)
            svr.serve_forever()

if __name__ == '__main__':

   tf.app.run()