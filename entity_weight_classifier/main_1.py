import tensorflow as tf
import os
from data_preprocess import Intent_Slot_Data
from model import embedding,sent_encoder,self_attention,loss_function,intent_acc,cosin_com,label_sent_attention,output_layers
import numpy as np
import logging
from sklearn.metrics import classification_report,precision_recall_fscore_support
import gc
import sys
sys.path.append('..')
from pre_data_deal.data_deal import Intent_Data_Deal
from xmlrpc.server import SimpleXMLRPCServer
from configparser import ConfigParser
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")
Config=ConfigParser()
Config.read('../Config.conf')
HOST=Config['host']['host']

from focal_loss import focal_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def get_sent_mask(sent_ids,entity_ids):

    sent_mask=np.zeros_like(sent_ids,dtype=np.float32)
    for i in range(sent_ids.shape[0]):
        for j in range(sent_ids.shape[1]):
            if sent_ids[i,j]>0 and sent_ids[i,j] not in entity_ids:
                sent_mask[i,j]=1.0
            elif sent_ids[i,j]>0 and sent_ids[i,j] in entity_ids:
                sent_mask[i,j]=2.0
    return sent_mask


def main(_):


    with tf.device('/gpu:1'):
        dd = Intent_Slot_Data(train_path="../dataset/train_out_char.txt",
                              test_path="../dataset/dev_out_char.txt",
                              dev_path="../dataset/dev_out_char.txt", batch_size=FLAGS.batch_size,
                              max_length=FLAGS.max_len, flag="train_new",
                              use_auto_bucket=FLAGS.use_auto_buckets,max_intent_length=FLAGS.max_label_len)
        id2intent = dd.id2intent
        label_word, label_len = dd.get_intent_word_array()
        intent_num=label_word.shape[0]
        intent_word_num=dd.intent_word_num
        word_num=dd.vocab_num

        sent_word=tf.placeholder(shape=(None,FLAGS.max_len),dtype=tf.int32)
        sent_len=tf.placeholder(shape=(None,),dtype=tf.int32)
        sent_mask=tf.placeholder(shape=(None,FLAGS.max_len),dtype=tf.float32)

        intent_y=tf.placeholder(shape=(None,intent_num),dtype=tf.int32)

        sent_emb=embedding(sent_word,word_num,FLAGS.embedding_dim,'sent_emb')
        label_emb=tf.Variable(tf.random_uniform(shape=(intent_num,300),dtype=tf.float32),trainable=True)
        sen_enc=sent_encoder(sent_word_emb=sent_emb,hidden_dim=FLAGS.hidden_dim,num=FLAGS.max_len,sequence_length=sent_len,name='sent_enc')
        sent_attention=self_attention(sen_enc,sent_mask)
        stack_sent_enc=tf.stack([ tf.concat((ele,sent_attention),1) for ele in sen_enc],1)
        # stack_sent_enc=sen_enc
        # stack_sent_enc=tf.stack(sen_enc,1)

        out=label_sent_attention(stack_sent_enc,label_emb,sent_mask)

        logit=output_layers(out,intent_num,name='out_layers',reuse=False)
        soft_logit=tf.nn.softmax(logit,1)


        class_y = tf.constant(name='class_y', shape=[intent_num, intent_num], dtype=tf.float32,
                              value=np.identity(intent_num), )

        logit_label=output_layers(label_emb,intent_num,name='out_layers',reuse=True)
        label_loss=tf.losses.softmax_cross_entropy(onehot_labels=class_y,logits=logit_label)

        loss=tf.losses.softmax_cross_entropy(onehot_labels=intent_y,logits=logit)

        loss=0.5*loss+0.5*label_loss

        # ss=tf.concat((sent_attention,tf.stack_sent_enc),1)
        # cosin=cosin_com(ss,label_emb,intent_num)

        # loss, soft_logit=loss_function(cosin,intent_y)

        # loss,soft_logit=loss_function(cosin,intent_y)
        # ss=tf.concat((sent_attention,sen_enc[0]),1)
        # logit=tf.layers.dense(ss,intent_num)
        # soft_logit=tf.nn.softmax(logit,1)
        # loss=focal_loss(soft_logit,tf.cast(intent_y,tf.float32))
        # # loss=tf.losses.softmax_cross_entropy(intent_y,logit)
        # #
        optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # opt=tf.train.AdamOptimizer(0.3)
        # grad_var=opt.compute_gradients(loss)
        # cappd_grad_varible=[[tf.clip_by_value(g,1e-5,1.0),v] for g,v in grad_var]
        # optimizer=opt.apply_gradients(grads_and_vars=cappd_grad_varible)


        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        with tf.device("/gpu:1"):
            _logger.info("load data")
            word_vocab = dd.vocab
            entity_id = []
            for k, v in word_vocab.items():
                if k in ['jb', 'bzxm', 'bxzl', 'sy', 'qj', 'bxcp']:
                    entity_id.append(v)

            if FLAGS.mod == 'train':
                with tf.Session(config=config) as sess:
                    num_batch = dd.num_batch
                    init_train_acc=0.0
                    init_dev_acc=0.0

                    saver=tf.train.Saver()
                    if os.path.exists('%s.meta'%FLAGS.model_dir):
                        saver.restore(sess,FLAGS.model_dir)
                    else:
                        sess.run(tf.global_variables_initializer())
                    dev_sent, dev_slot, dev_intent, dev_rel_len = dd.get_dev()
                    dev_sent_mask = get_sent_mask(dev_sent, entity_id)
                    train_sent, train_slot, train_intent, train_rel_len = dd.get_train()
                    train_sent_mask = get_sent_mask(train_sent, entity_id)
                    for i in range(FLAGS.epoch):
                        for _ in range(num_batch):
                            sent, slot, intent_label, rel_len, cur_len = dd.next_batch()
                            batch_sent_mask = get_sent_mask(sent, entity_id)

                            soft_logit_, loss_, _ = sess.run([soft_logit, loss, optimizer], feed_dict={sent_word: sent,
                                                                                                       sent_len: rel_len,
                                                                                                       intent_y: intent_label,
                                                                                                       sent_mask: batch_sent_mask
                                                                                                       })
                        train_soft_logit_, train_loss_ = sess.run([soft_logit, loss], feed_dict={sent_word: train_sent,
                                                                                                 sent_len: train_rel_len,
                                                                                                 intent_y: train_intent,
                                                                                                 sent_mask: train_sent_mask
                                                                                                 })
                        train_acc = intent_acc(train_soft_logit_, train_intent,id2intent)

                        dev_soft_logit_, dev_loss_ = sess.run([soft_logit, loss],
                                                              feed_dict={sent_word: dev_sent,
                                                                         sent_len: dev_rel_len,
                                                                         intent_y: dev_intent,
                                                                         sent_mask: dev_sent_mask
                                                                         })
                        dev_acc = intent_acc(dev_soft_logit_, dev_intent,id2intent)

                        if train_acc>init_train_acc and dev_acc>init_dev_acc:
                            init_train_acc=train_acc
                            init_dev_acc=dev_acc
                            _logger.info('save')
                            saver.save(sess,FLAGS.model_dir)

                        _logger.info('第 %s 次迭代  train_loss:%s train_acc:%s dev_loss:%s dev_acc:%s'%(i,train_loss_, train_acc, dev_loss_, dev_acc))

            elif FLAGS.mod == 'server':
                with tf.Session(config=config) as sess:

                    saver = tf.train.Saver()
                    sess = tf.Session(config=config)
                    if os.path.exists('%s.meta' % FLAGS.model_dir):
                        saver.restore(sess, '%s' % FLAGS.model_dir)
                    else:
                        _logger.error('lstm没有模型')

                    idd = Intent_Data_Deal()

                    def intent(sent_list):
                        sents = []
                        _logger.info("%s" % len(sent_list))

                        sents=[]
                        for sent in sent_list:

                            sents.append(idd.deal_sent(sent))

                        sent_arr, sent_vec = dd.get_sent_char(sents)
                        infer_sent_mask=get_sent_mask(sent_arr, entity_id)

                        intent_logit = sess.run(soft_logit, feed_dict={sent_word: sent_arr,
                                                                    sent_len: sent_vec,
                                                                       sent_mask:infer_sent_mask})

                        res = []
                        for ele in intent_logit:
                            ss = [[id2intent[index], str(e)] for index, e in enumerate(ele) if e >= 0.3]
                            if not ss:
                                ss = [[id2intent[np.argmax(ele)], str(np.max(ele))]]
                            res.append(ss)

                        del sents
                        gc.collect()
                        _logger.info('process end')
                        _logger.info('%s'%res)
                        return res

                    svr = SimpleXMLRPCServer((HOST, 8087), allow_none=True)
                    svr.register_function(intent)
                    svr.serve_forever()






if __name__ == '__main__':
    tf.app.run()