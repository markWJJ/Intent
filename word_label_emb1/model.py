import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


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
        embedding=tf.Variable(tf.random_uniform(shape=(num,emb_dim),minval=-1.0,maxval=1.0,dtype=tf.float32),name=name,trainable=False)
        emb=tf.nn.embedding_lookup(embedding,sent)
        return emb,embedding



def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, num_class):

    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    # x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    x_emb_0=tf.cast(x_emb,tf.float32)
    x_mask=tf.cast(x_mask,tf.float32)

    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2)  # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c
    x_full_emb = x_emb_0
    Att_v = tf.contrib.layers.conv2d(G, num_outputs=num_class, kernel_size=[55], padding='SAME',
                                     activation_fn=tf.nn.relu)  # b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
    Att_v_max = tf.clip_by_value(partial_softmax(Att_v, x_mask, 1, 'Att_v_max'),1e-5,1.0)  # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max) #b*s*e
    H_enc = tf.reduce_sum(x_att, axis=1)# b*e
    return H_enc


def discriminator_1layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    return H_dis


def discriminator_0layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    logits = layers.linear(tf.nn.dropout(H, keep_prob=dropout), num_outputs=num_outputs, biases_initializer=biasInit,
                           scope=prefix + 'dis', reuse=is_reuse)
    return logits


def linear_layer(x, output_dim):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, hidden_dim, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=hidden_dim,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits


def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis = layers.fully_connected(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_3', reuse=is_reuse)
    return logits


def partial_softmax(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True) #b*1*1
        partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)#b*s*1

        return partial_softmax_score

