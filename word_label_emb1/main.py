import tensorflow as tf
import sys
sys.path.append('.')
import os
from data_preprocess import Intent_Slot_Data
from model import embedding,att_emb_ngram_encoder_maxout,discriminator_2layer
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 128
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




def main(_):

    dd = Intent_Slot_Data(train_path="../dataset/train_out_char.txt",
                          test_path="../dataset/dev_out_char.txt",
                          dev_path="../dataset/dev_out_char.txt", batch_size=FLAGS.batch_size,
                          max_length=FLAGS.max_len, flag="train_new",
                          use_auto_bucket=FLAGS.use_auto_buckets,max_intent_length=FLAGS.max_label_len)

    label_word, label_len = dd.get_intent_word_array()
    intent_num=label_word.shape[0]
    intent_word_num=dd.intent_word_num
    word_num=dd.vocab_num
    with tf.device('/gpu:1'):
        sent_word=tf.placeholder(shape=(None,FLAGS.max_len),dtype=tf.int32)
        sent_len=tf.placeholder(shape=(None,),dtype=tf.int32)
        sent_mask=tf.placeholder(shape=(None,FLAGS.max_len),dtype=tf.int32)

        intent_y=tf.placeholder(shape=(None,intent_num),dtype=tf.int32)

        sent_emb,sent_W=embedding(sent_word,word_num,FLAGS.embedding_dim,'sent_emb')
        label_pos=tf.argmax(intent_y,-1)
        label_emb,label_w=embedding(label_pos,intent_num,FLAGS.embedding_dim,'label_emb')

        sent_emb=tf.cast(sent_emb,tf.float32)

        H_enc=att_emb_ngram_encoder_maxout(sent_emb,sent_mask,label_emb,tf.transpose(label_w,[1,0]),intent_num)

        logit=discriminator_2layer(H_enc,hidden_dim=300,dropout=0.9,prefix='classify',num_outputs=intent_num,is_reuse=False)
        logit_class=discriminator_2layer(label_w,hidden_dim=300,dropout=0.9,prefix='classify',num_outputs=intent_num,is_reuse=True)

        prob = tf.nn.softmax(logit)
        class_y = tf.constant(name='class_y', shape=[intent_num, intent_num], dtype=tf.float32,
                              value=np.identity(intent_num), )

        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(intent_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=intent_y, logits=logit)) + 0.2 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logit_class))

        # loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=intent_y, logits=tf.clip_by_value(logit,1e-5,1.0)))

        global_step = tf.Variable(0, trainable=False)

        # opt=tf.train.AdamOptimizer(0.001)
        # grad_var=opt.compute_gradients(loss)
        # cappd_grad_varible=[[tf.clip_by_value(g,1e-5,1.0),v] for g,v in grad_var]
        # train_op=opt.apply_gradients(grads_and_vars=cappd_grad_varible)

        train_op=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # train_op = layers.optimize_loss(
        #     loss,
        #     global_step=global_step,
        #     optimizer=tf.train.AdamOptimizer,
        #     learning_rate=FLAGS.learning_rate)


        # label_emb=embedding(label_word,intent_word_num,FLAGS.embedding_dim,'intent_emb')

        # sen_enc=sent_encoder(sent_word_emb=sent_emb,hidden_dim=FLAGS.hidden_dim,num=FLAGS.max_len,sequence_length=sent_len,name='sent_enc')
        #
        # label_enc=sent_encoder(sent_word_emb=label_emb,hidden_dim=FLAGS.hidden_dim,num=FLAGS.max_label_len,sequence_length=label_len,name='label_enc')
        # cosin=cosin_com(sen_enc,label_enc,intent_num)
        #
        # loss,soft_logit=loss_function(cosin,intent_y)
        # # logit=tf.layers.dense(sen_enc[0],intent_num)
        # # print(logit)
        # # soft_logit=tf.nn.softmax(logit,1)
        # # loss=tf.losses.softmax_cross_entropy(intent_y,logit)
        # #
        # optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)
        # #
        # opt=tf.train.AdamOptimizer(0.3)
        # grad_var=opt.compute_gradients(loss)
        # cappd_grad_varible=[[tf.clip_by_value(g,1e-5,1.0),v] for g,v in grad_var]
        # optimizer=opt.apply_gradients(grads_and_vars=cappd_grad_varible)

    config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            log_device_placement=False,
                            allow_soft_placement=True,)

    with tf.Session(config=config) as sess:
        num_batch=dd.num_batch

        sess.run(tf.global_variables_initializer())
        train_sent, train_slot, train_intent, train_rel_len = dd.get_train()
        dev_sent, dev_slot, dev_intent, dev_rel_len = dd.get_dev()

        ss = []
        for ele in train_rel_len:
            s = [1] * ele
            s.extend([0] * (FLAGS.max_len - ele))
            ss.append(s)
        train_X_mask = np.array(ss)


        dev_ss = []
        for ele in dev_rel_len:
            s = [1] * ele
            s.extend([0] * (FLAGS.max_len - ele))
            dev_ss.append(s)
        dev_X_mask = np.array(dev_ss)

        for _ in range(300):

            for _ in range(num_batch):

                sent, slot, intent_label, rel_len, cur_len = dd.next_batch()

                ss=[]
                for ele in rel_len:
                    s=[1]*ele
                    s.extend([0]*(FLAGS.max_len-ele))
                    ss.append(s)
                X_mask=np.array(ss)
                s_,h_,_,loss_,step,acc=sess.run([sent_emb,H_enc,train_op,loss,global_step,accuracy],feed_dict={sent_word:sent,
                                            sent_len:rel_len,
                                            intent_y:intent_label,
                                                sent_mask:X_mask
                                            })
                # print('-'*30,loss_,acc,h_[0][0],s_[0][0][0])
            train_acc, train_loss_= sess.run([accuracy, loss],
                                         feed_dict={sent_word: train_sent,
                                                    sent_len:train_rel_len,
                                                    intent_y:train_intent,
                                                    sent_mask:train_X_mask
                                                    })

            dev_acc, dev_loss_ = sess.run([accuracy, loss],
                                        feed_dict={sent_word: dev_sent,
                                                   sent_len: dev_rel_len,
                                                   intent_y: dev_intent,
                                                   sent_mask: dev_X_mask
                                                   })
            print('train_loss:%s train_acc:%s dev_loss:%s dev_acc:%s'%(train_loss_,train_acc,dev_loss_,dev_acc))






if __name__ == '__main__':
    tf.app.run()