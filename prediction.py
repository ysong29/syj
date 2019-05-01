from __future__ import print_function
import tensorflow as tf
import pandas as pd
######load model
sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.import_meta_graph('save/nets-10000.meta')
saver.restore(sess, "save/nets-10000")
graph=tf.get_default_graph()
input=graph.get_tensor_by_name('input:0')
w1=graph.get_tensor_by_name('w0:0')
b1=graph.get_tensor_by_name('b0:0')
w2=graph.get_tensor_by_name('w:0')
b2=graph.get_tensor_by_name('b:0')
output=tf.matmul(tf.matmul(input, w1)+b1,w2)+b2
##########make prediction
pred_data=pd.read_csv('1.csv')
pred_data1 = pred_data.iloc[0:5,2:13]
pred_norm=(pred_data1-pred_data1.min())/(pred_data1.max()-pred_data1.min())
print('prediction: ')
print(sess.run(output,feed_dict={input:pred_norm}))
