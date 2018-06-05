import tensorflow as tf
import os
from word_label_emb.data_preprocess import Intent_Slot_Data
import numpy as np

class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 16
    label_max_len=16
    sent_len = 30  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 100
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = './save_model/model_lstm/intent_lstm.ckpt'
    if not os.path.exists('./save_model/model_lstm'):
        os.makedirs('./save_model/model_lstm')
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
tf.app.flags.DEFINE_integer("max_label_len", config.label_max_len, "句子长度")
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

    def __init__(self):
        dd = Intent_Slot_Data(train_path="../dataset/train_out_char.txt",
                              test_path="../dataset/dev_out_char.txt",
                              dev_path="../dataset/dev_out_char.txt", batch_size=FLAGS.batch_size,
                              max_length=FLAGS.max_len, flag="train_new",
                              use_auto_bucket=FLAGS.use_auto_buckets, max_intent_length=FLAGS.max_label_len)
        self.dd=dd
        label_word, label_len = dd.get_intent_word_array()
        intent_num = label_word.shape[0]
        intent_word_num = dd.intent_word_num
        word_num = dd.vocab_num

        self.sent_word = tf.placeholder(shape=(None, FLAGS.max_len), dtype=tf.int32)
        self.sent_len = tf.placeholder(shape=(None,), dtype=tf.int32)

        self.intent_y = tf.placeholder(shape=(None, intent_num), dtype=tf.int32)

        # embedding = tf.Variable(tf.random_uniform(shape=(word_num, 100), dtype=tf.float32), name='sent_emb')
        # self.sent_emb = tf.nn.embedding_lookup(embedding, self.sent_word)

        self.sent_emb = self.embedding(self.sent_word, word_num, FLAGS.embedding_dim, 'sent_emb')
        self.label_emb = self.embedding(label_word, intent_word_num, FLAGS.embedding_dim, 'intent_emb')
        self.sent_enc = self.sent_encoder(sent_word_emb=self.sent_emb, hidden_dim=FLAGS.hidden_dim, num=FLAGS.max_len,
                               sequence_length=self.sent_len, name='sent_enc')

        self.label_enc = self.sent_encoder(sent_word_emb=self.label_emb, hidden_dim=FLAGS.hidden_dim, num=FLAGS.max_label_len,
                                 sequence_length=label_len, name='label_enc')


        # self.sent_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(self.sent_enc[0]), axis=1))

        self.cosin = self.cosin_com(self.sent_enc, self.label_enc, intent_num)

        self.loss = self.loss_function(self.cosin, self.intent_y)
        opt = tf.train.AdamOptimizer(0.0001)
        grad_var = opt.compute_gradients(self.loss)
        cappd_grad_varible = [[tf.clip_by_value(g, 1e-5, 1.0), v] for g, v in grad_var]
        self.optimizer = opt.apply_gradients(grads_and_vars=cappd_grad_varible)

    def embedding(self,sent,num,emb_dim,name):
        '''
        词嵌入
        :param sent:
        :param num:
        :param emb_dim:
        :param name:
        :return:
        '''
        with tf.variable_scope(name_or_scope=name):
            embedding=tf.Variable(tf.random_uniform(shape=(num,emb_dim),dtype=tf.float32),name='sent_emb')
            emb=tf.nn.embedding_lookup(embedding,sent)
            return emb


    def sent_encoder(self,sent_word_emb,num,hidden_dim,sequence_length,name):
        '''
        句编码
        :param sent_word_emb:
        :param hidden_dim:
        :param name:
        :return:
        '''
        with tf.variable_scope(name_or_scope=name):
            sent_word_embs=tf.unstack(sent_word_emb,num,1)
            lstm_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
            lstm_cell_1=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
            encoder,_=tf.nn.bidirectional_dynamic_rnn(
                lstm_cell,
                lstm_cell_1,
                sent_word_emb,
                dtype=tf.float32,
                sequence_length=sequence_length, )
            # encoder,_=tf.nn.static_rnn(lstm_cell,sent_word_embs,sequence_length=sequence_length,dtype=tf.float32)
            encoder=tf.concat(encoder,2)
            encoder=tf.stack(encoder,0)
            encoder=tf.layers.dense(encoder,100)
            encoder=tf.unstack(encoder,num,1)
            return encoder


    def cosin_com(self,sent_enc,label_enc,label_num):
        '''
        相似度计算
        :param sent_enc:
        :param label_enc:
        :return:
        '''
        sent=sent_enc[0]
        label=label_enc[0]
        sent_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(sent), axis=1))
        label=tf.unstack(label,label_num,0)
        cosins = []
        # 内积
        for ele in label:
            intent_norm = tf.sqrt(tf.reduce_sum(tf.square(ele)))
            ele = tf.expand_dims(ele, -1)
            sent_intent = tf.matmul(sent, ele)
            sent_intent = tf.reshape(sent_intent, [-1, ])
            cosin = sent_intent / (sent_emb_norm * intent_norm)
            cosins.append(cosin)
        cosin = tf.stack(cosins, 1)

        return cosin

    def loss_function(self,cosin,label):
        soft_logit = tf.nn.softmax(cosin, 1)
        intent = tf.cast(label, tf.float32)
        intent_loss = -tf.reduce_sum(intent * tf.log(tf.clip_by_value(soft_logit, 1e-5, 1.0, name=None)))
        return intent_loss

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(100):
                sent, slot, intent_label, rel_len, cur_len = self.dd.next_batch()
                loss_,_= sess.run([self.loss,self.optimizer], feed_dict={self.sent_word: sent,
                                                                        self.sent_len: rel_len,
                                                                        self.intent_y: intent_label
                                                                                             })
                print(loss_)
def main(_):
    model=Model()
    model.train()

if __name__ == '__main__':
    tf.app.run()

