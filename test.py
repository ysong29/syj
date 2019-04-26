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
#######test
#####reshape data to get inputdata of test
#1
test=pd.read_csv('14.csv')
test1 = test.iloc[0:10,2:13]
test_norm=(test1-test1.min())/(test1.max()-test1.min())
test_mvp= test_norm[['Share']]
test_input=test_norm.iloc[0:10,0:10]
####begain test1
print('result1: ')
print(sess.run(output,feed_dict={input:test_input}))
#2
test15=pd.read_csv('15.csv')
test15_reshape = test15.iloc[0:12,2:13]
test15_norm=(test15_reshape-test15_reshape.min())/(test15_reshape.max()-test15_reshape.min())
test15_mvp= test15_norm[['Share']]
test15_input=test15_norm.iloc[0:12,0:10]
#######begin test2
print('result2: ')
print(sess.run(output,feed_dict={input:test15_input}))
#3
test16=pd.read_csv('16.csv')
test16_reshape = test16.iloc[0:12,2:13]
test16_norm=(test16_reshape-test16_reshape.min())/(test16_reshape.max()-test16_reshape.min())
test16_mvp= test16_norm[['Share']]
test16_input=test16_norm.iloc[0:12,0:10]
#######begin test3
print('result3: ')
print(sess.run(output,feed_dict={input:test16_input}))
#4
test17=pd.read_csv('17.csv')
test17_reshape = test17.iloc[0:10,2:13]
test17_norm=(test17_reshape-test17_reshape.min())/(test17_reshape.max()-test17_reshape.min())
test17_mvp= test17_norm[['Share']]
test17_input=test17_norm.iloc[0:10,0:10]
#######begin test4
print('result4: ')
print(sess.run(output,feed_dict={input:test17_input}))
#5
test18=pd.read_csv('18.csv')
test18_reshape = test18.iloc[0:10,2:13]
test18_norm=(test18_reshape-test18_reshape.min())/(test18_reshape.max()-test18_reshape.min())
test18_mvp= test18_norm[['Share']]
test18_input=test18_norm.iloc[0:10,0:10]
#######begin test5
print('result5: ')
print(sess.run(output,feed_dict={input:test18_input}))
