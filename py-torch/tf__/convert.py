import tensorflow as tf

# 卷积层设计
# -------------------------------------------------------------------------------------------------
# 3为当前的深度，16为通过之后的深度
filter_weight = tf.get_variable('weight', [5,5,3,16], initializer = tf.truncated_normal_initializer(stddev = 0.1))
biases = tf.get_variable("biases", [16], initializer = tf.truncated_normal_initializer(stddev = 0.1))

# conv2d 提供了方便的函数了解前向传播算法
# 维度分别为  batch  卷积层权重  stride  padding
conv = tf.nn.conv2d(input, filter_weight, strides = [1, 1, 1, 1], padding = 'SAME')
bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)


# 池化层设计
# ---------------------------------------------------------------------------------------------------
# 具体参数  计算的参数，尺寸，步长，padding
pool = tf.nn.max_pool(actived_conv, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding ='SAME')