# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf



def add_continual_fc_layer(inputs, in_size, out_size, add_num, name_prefix=''):
    """
    :param inputs: 二维， bs * d
    :param in_size: 输入的维度：d
    :param out_size: 输出的维度
    :return: bs * out_size
    """
    if add_num < 0 or add_num >= out_size:
        print("ERROR: add num error.")
        return None
    wlimit = np.sqrt(6 / (in_size + (out_size - add_num)))
    W_old = tf.get_variable(name_prefix+'W_old', shape=[in_size, out_size - add_num],
                            initializer=tf.random_uniform_initializer(-wlimit, wlimit))
    b_old = tf.get_variable(name_prefix+'b_old', shape=[out_size - add_num],
                            initializer=tf.random_uniform_initializer(-wlimit, wlimit))
    if add_num:
        wlimit = np.sqrt(6 / (in_size + add_num))
        W_new = tf.get_variable(name_prefix+'new_W', shape=[in_size, add_num],
                                initializer=tf.random_uniform_initializer(-wlimit, wlimit))
        W = tf.identity(tf.concat([W_old, W_new], axis=-1), name=name_prefix+'W')
        b_new = tf.get_variable(name_prefix+'new_b', shape=[add_num],
                                initializer=tf.random_uniform_initializer(-wlimit, wlimit))
        b = tf.identity(tf.concat([b_old, b_new], axis=-1), name=name_prefix+'b')

    else:
        W = tf.identity(W_old, name=name_prefix+'W')
        b = tf.identity(b_old, name=name_prefix+'b')
    # b = tf.get_variable('b', shape=[out_size],
    #                     initializer=tf.random_uniform_initializer(-wlimit, wlimit))

    Wx_plus_b = tf.matmul(inputs, W) + b

    return Wx_plus_b


def add_continual_fc_layer_serial(inputs, in_size, out_size, accumulate_add_new_params_list, index=1):
    """
    :param inputs: 二维， bs * d
    :param in_size: 输入的维度：d
    :param out_size: 输出的维度
    :param accumulate_add_new_params_list: 是每次新增的维度，第一维是第一次新增维度，最后一次是本次要新增的维度 [['new_', 24,3,12,12],['batch_2', 12,2,4,4]]
    :param index 是使用accumulate_add_new_params_list中的第几维的参数进行扩展
    :return: bs * out_size
    """
    first_num = out_size  # 用于记录最初的维度
    print('before first_num=')
    print(first_num)
    for ele in accumulate_add_new_params_list:
        # prefix_name, tag_add_num, action_add_num, cate_add_num_0, cate_add_num_1 = ele
        add_num = int(ele[index])
        first_num -= int(add_num)
        if int(add_num) < 0 or int(add_num) >= out_size:
            print("ERROR: add num error.")
            return None
    print('after first_num=')
    print(first_num)

    wlimit = np.sqrt(6.0 / (in_size + first_num))
    W_old = tf.get_variable('W_old', shape=[in_size, first_num],
                            initializer=tf.random_uniform_initializer(-wlimit, wlimit))
    b_old = tf.get_variable('b_old', shape=[first_num],
                            initializer=tf.random_uniform_initializer(-wlimit, wlimit))

    if len(accumulate_add_new_params_list) > 0:
        for ele in accumulate_add_new_params_list:
            # prefix_name, tag_add_num, action_add_num, cate_add_num_0, cate_add_num_1 = ele
            name_prefix, add_num = ele[0], int(ele[index])
            print('[in here], add_num=%d' % add_num)
            wlimit = np.sqrt(6.0 / (in_size + add_num))
            W_new = tf.get_variable(name_prefix+'cateWeight', shape=[in_size, add_num],
                                    initializer=tf.random_uniform_initializer(-wlimit, wlimit))
            W_old = tf.identity(tf.concat([W_old, W_new], axis=-1), name='W')
            b_new = tf.get_variable(name_prefix+'catebias', shape=[add_num],
                                    initializer=tf.random_uniform_initializer(-wlimit, wlimit))
            b_old = tf.identity(tf.concat([b_old, b_new], axis=-1), name='b')
        W = W_old
        b = b_old
    else:
        W = tf.identity(W_old, name='W')
        b = tf.identity(b_old, name='b')
    # b = tf.get_variable('b', shape=[out_size],
    #                     initializer=tf.random_uniform_initializer(-wlimit, wlimit))

    Wx_plus_b = tf.matmul(inputs, W) + b

    return Wx_plus_b