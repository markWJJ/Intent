import tensorflow as tf
import os
import numpy as np


def embedding(sent,num,emb_dim,name):
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


def sent_encoder(sent_word_emb,num,hidden_dim,sequence_length,name):
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
        encoder=tf.unstack(encoder,num,1)
        print(encoder)
        # encoder=tf.layers.dense(encoder,100,activation=tf.nn.tanh)
        # encoder=tf.unstack(encoder,num,1)
        return encoder


def cosin_com(sent_enc,label_enc,label_num):
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

def loss_function(cosin,label):
    soft_logit = tf.nn.softmax(cosin, 1)
    intent = tf.cast(label, tf.float32)
    intent_loss = -tf.reduce_sum(intent * tf.log(tf.clip_by_value(soft_logit, 1e-5, 1.0, name=None)))
    return intent_loss,soft_logit

def intent_acc(pre,label):
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

    return float(num) / float(all_sum)


