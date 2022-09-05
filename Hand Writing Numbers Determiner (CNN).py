import tensorflow._api.v2.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.disable_v2_behavior()

mnist = input_data.read_data_sets('data/', one_hot=True)

x = tf.placeholder("float", shape=[None, 28, 28, 1])
y_ = tf.placeholder("float", shape=[None, 10])
# 在卷积神经网络中，所有数据格式都是思维的，[batchsize, h, w, c]
# batchsize: 表示一次迭代的样本数量
# h：表示图像的长度
# w：表示图像的宽度
# c：表示颜色通道(或者特征图的个数)
# 需要注意h和w的顺寻，不同深度学习框架先后顺序可能并不一样。在placeholder()中对输入数据也需要进行明确的定义，标签与之前一样

# 接下来就是权重参数初始化，由于时卷积操作，所以它与之前全连接定义方式完全不同
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
# 对卷积核进行随机初始化操作，w表示卷积核，b表示偏置。其中需要指定的就是卷积核的大小，同样也是四维的。
# [5, 5, 1, 32]表示使用卷积核的大小是5*5， 前面连接的输入颜色通道时1（如果是特征图，就是特征图个数)， 使用卷积核的个数是32， 就是通过卷积后
# 得到32个特征图。卷积层中需要设置的参数稍微有点多，需要注意卷积核的深度，一定要与前面的输入深度一致
# 对于偏置参数来说，方法还是相同的，只需要看最终结果。卷积中设置了32个卷积核，那么肯定会有32个特征图，偏置参数相当于要对每一个特征图上的数值进行
# 微调，所以个数是32

# 卷积的计算流程
h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Tensorflow中已经实现卷积操作，使用conv2d函数即可，需要传入当前的输入数据，卷积核参数，步长以及padding项
# 步长同样是四维的，第一个维度中，1表示在batchsize上滑动，通常情况下都是1，表示一个一个样本轮着来。第二个和第三个表示在图像上的长度和宽度上的滑动
# 都是每次一个单位，可以看成一个小整体[1, 1]，长度和宽度的滑动一般都是一致的，如果是[2, 2]，表示移动两个单位。最后一个1表示在颜色通道或者特征图
# 上移动，基本也是1.通常情况下，步长参数在图像任务中只需要按照网络的涉及改动中间数值，如果应用到其他领域，就需要具体分析
# padding中可以设置是否加入填充，这里指定成SAME，表示需要加入padding项。在池化层中，还需要指定ksize，也就是依次选择的区域大小，与卷积核参数类
# 似，只不过这里没有参数计算。[1, 2, 2, 1]与步长的参数含义一致，通常batchsize和通道上都为1，只需要改变中间的[2,2]来控制池化层结果，这里选择
# ksize和stride都是2，相当于长和宽各压缩到原来的一半

# 第一层确定后，后续的卷积和池化操作也相同，继续进行叠加就行。
# 一般来说，都是按照相同的方式进行叠加的，所以可以先定义好组合函数


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 对于Mnist数据集来说，用两个卷积层就差不多

# 使用全连接层来组合已经提取出的特征
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 这里需要定义好全连接的权重参数：[7 * 7 * 64, 1024], 全连接参数与卷积参数有些不同，此时需要一个二维的矩阵参数。第二个维度1024表示要把卷积提取
# 特征图转换成1024维的特征。第一个维度需要自己计算，也就是当前输入特征图的大小，Mnist本身输入的28 * 28 * 1，给定上述参数后，经过卷积后的大小保
# 持不变，池化操作后，长度和宽度都变为原来的一般，代码中选择两个池化操作，所以最终的特征图大小为28 * 1/2 * 1/2 = 7. 特征图个数是由最后一次卷积
# 操作决定，也就是64. 这样就可以把7 * 7 * 64这个参数计算出来
# 在全连接操作前，需要reshape一下特征图，也就是将一个特征图压扁或拉长成为一个特征，最后进行矩阵乘法，就完成了全连接的工作

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 由于会有潜在的过拟合问题，此时可以加进dropout项，在进行全连接是，表示希望保存神经元的百分比，例如50%

# 第二个全连接
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# 最终结果是一个十分类的手写字体识别，第二个全连接的目的就是把特征转换成最终的结果

# 损失函数
crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 这里使用了AdamOptimizer()优化器，相当于在学习的过程中，让学习率逐渐减少，符合实际要求，而且将计算准确率作为衡量标准

# 运行
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batchSize = 50
for i in range(1000):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize, 28, 28, 1])
    trainingLabels = batch[1]
    trainingAccuracy = accuracy.eval(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, trainingAccuracy))
    trainStep.run(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 0.5})
