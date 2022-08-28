import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.disable_v2_behavior()

mnist = input_data.read_data_sets('data/', one_hot=True)
print("类型是 %s" % (type(mnist)))
print("训练数据有 %d" % (mnist.train.num_examples))
print("测试数据有 %d" % (mnist.test.num_examples))

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
# 28 * 28 * 1
print("数据类型 is %s" % (type(trainimg))) # ndarray type
print("标签类型 %s" % (type(trainlabel))) # ndarray type
print("训练集的shape %s" % (trainimg.shape,)) # (55000, 784) 代表有55000个训练数据，每个数据有784个特征
print("训练集标签的shape %s" % (trainlabel.shape,)) # (55000, 10) 代表有55000个训练标签，每个数据有10个label
print("测试集的shape %s" % (testlabel.shape,)) # (10000， 784) 代表有10000个测试数据，每个数据有784个特征
print("测试集标签的shape %s" % (testlabel.shape,)) # (10000， 10) 代表有10000个测试标签，每个数据有10个特征
# 784是因为所有的图片都有784个像素点，每个像素点都是一个特征
# 10采用的是独热编码，代表了每个图片的结果。如果一张图像是3这个数字，标签就是[0,0,0,1,0,0,0,0,0,0]

# 随机展示将一组数据转换成实际的图片
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample) # 随机抽取一组数据
random_value = np.random.choice(randidx, size=None, replace=True, p=None) # 随机从上组数据抽取其中一个数据
curr_img = np.reshape(trainimg[random_value, :], (28,28)) # 将那一行转换成28 * 28的矩阵
curr_label = np.argmax(trainlabel[random_value, :]) # argmax返回最大的值的index
plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
print("" + str(random_value) + "th 训练数据 " + " 标签是 " + str(curr_label))
plt.show()

# 开始搭建神经网络
numClasses = 10 # 代表所有数据都是用于完成十分类的任务
inputSize = 784 # 代表初始输入值是784个，每个输入数据大小都是一模一样的
numHiddenUnits = 64 # 代表隐藏神经元的数量是64个
numHiddenUnitsLayer2 = 100
trainingIterations = 10000 # 代表梯度优化会执行10000次
batchSize = 64

X = tf.placeholder(tf.float32, shape= [None, inputSize])
y = tf.placeholder(tf.float32, shape= [None, numClasses])
# placeholder 是用于输入和输出的入口和出口，在运行前定义后，运行后数据会通过他们输入进去
# shape中None代表不限制batch的大小，一次可以迭代多个数据. inputSize表示会有多少个初值batch进入，numClass表示有多少个值输出

# 开始初始化每层的之间的权重和偏置
W1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numHiddenUnitsLayer2], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numHiddenUnitsLayer2])
W3 = tf.Variable(tf.truncated_normal([numHiddenUnitsLayer2, numClasses], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1), [numClasses])
# 这里的权重参数使用高斯初始化， 并且控制其范围在较小的范围浮动，用tf.truncated_normal函数对随机结果进行限制.
# 在高斯化的情况下，如果输入参数为mean = 0, stddev = 1， 就不太可能出现[-2. 2]的点，相当于截断标准是2倍的stddev
# [inputSize, numHiddenUnits]代表权重矩阵是这个大小，可以直接与前面一层相乘
# 对于偏置参数，用常数来复制就可，但是要注意其个数要与输出结果一致

# 开始计算
hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput) # 重要
hiddenLayer2Output = tf.matmul(hiddenLayerOutput, W2) + B2
hiddenLayer2Output = tf.nn.relu(hiddenLayer2Output)
finalOutput = tf.matmul(hiddenLayer2Output, W3) + B3
# 使用matmul进行矩阵乘法并且加上偏置
# 使用激活函数对值进行激活
# 再次使用矩阵乘法得出最终值

# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# 使用对数函数计算损失函数，然后使用梯度下降进行优化

# 计算准确率
correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 观察预测的最大位置和实际的最大位置是否一致

# 开始执行
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _, trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i % 1000 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print("step %d, training accuracy %g" % (i, trainAccuracy))

testInputs = mnist.test.images
testLabels = mnist.test.labels
acc = accuracy.eval(session=sess, feed_dict={X: testInputs, y: testLabels})
print("testing accuracy: {}".format(acc))
