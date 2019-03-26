import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
train_X = np.linspace(-1, 1, 100)
train_Y = 2*train_X + np.random.randn(*train_X.shape)*0.3

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

Weights = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
z = tf.multiply(X,Weights)+ b
plt.plot(train_X,train_Y, 'ro', label = 'original data')



cost = tf.reduce_mean(tf.square(Y-z))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

training_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    plotdata = {'batchsize':[],'loss': []}
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            if epoch % display_step == 0:
                lost = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
                print("Epoch:",epoch+1,'cost:',lost,"W=",sess.run(Weights),'B=',sess.run(b))
                if not (lost == "NA"):
                    plotdata['batchsize'].append(epoch)
                    plotdata['loss'].append(lost)

    print('Finished')
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(Weights),'B=',sess.run(b))
    plt.plot(train_X,sess.run(Weights)*train_X+sess.run(b),label = 'fittedline')
    plt.legend()
    plt.grid()
    plt.show()






