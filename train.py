from __future__ import print_function
import tensorflow as tf
import pandas as pd
#read data
data = pd.read_csv('1-14.csv')
data1 = data.iloc[0:134,2:13]
#data normalizaion
data_norm=(data1-data1.min())/(data1.max()-data1.min())
# make result
mvp_share= data_norm[['Share']]
#reshape data
data_new=data_norm.iloc[0:134,0:10]

#define input and output data
input_data = tf.placeholder(tf.float32,[None,10],name='input')
output_data = tf.placeholder(tf.float32,[None,1],name='output')
##### construct nerual network layer(3 layers total)
#define input layer
def input_layer(init_input,input_size0,output_size0):

    w0=tf.Variable(tf.random_normal([input_size0,output_size0]),name='w0')####define weights of input data
    b0 = tf.Variable(tf.zeros([output_size0])+0.1,name='b0')#######define bias of input data
    input0= tf.matmul(tf.cast(init_input,tf.float32),w0)+b0##########no activation function needed in input
    return input0
#define hidden layer
def hidden_layer(input0, input_size, output_size):
    input1=tf.sigmoid(input0)
    w=tf.Variable(tf.random_normal([input_size,output_size]),name='w')########define weights for hidden layer
    b=tf.Variable(tf.zeros([output_size])+0.1,name='b')########define bias for hidden layer
    input=tf.matmul(input1, w)+b##############use sigmoid as activation function
    return input
#define output layer
def output_layer(input):
    output=tf.sigmoid(input)
    return output
#input data
l_input=input_layer(input_data,10,30)
l_hidden=hidden_layer(l_input,30,1)
l_output=output_layer(l_hidden)
#calculate error
loss = tf.losses.mean_squared_error(output_data, l_output)
train_optimizer= tf.train.AdamOptimizer(0.001).minimize(loss)
#initialize variable
init = tf.global_variables_initializer()
######save model
saver = tf.train.Saver(max_to_keep=1)
sess=tf.Session()
sess.run(init)

#traning
for i in range(10000):
    sess.run(train_optimizer,feed_dict={input_data : data_new, output_data : mvp_share})
    if i%1000==0:
        print(sess.run(loss, feed_dict={input_data:data_new, output_data : mvp_share}))
saver.save(sess, 'save/nets',global_step=10000)
print('save model succeed')