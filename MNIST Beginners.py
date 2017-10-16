import tensorflow as tf
# Importing MNIST DataSets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Initializing the regression variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))  # Wight
b = tf.Variable(tf.zeros([10]))  # Bias
# Softmax Function apply
y = tf.nn.softmax(tf.matmul(x, W) + b)  # Predicting Probabilities
y_ = tf.placeholder(tf.float32, [None, 10])  # True distribu

# Training Section
# Cross Entropy Function to measure the inefficient predictions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Optimizing the result
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Creating a Session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train Loop
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating Model
correct_prediction = tf.equal(tf.argmax(y, 1),
                              tf.argmax(y_, 1))  # Checking the predictions if true Returns List of Booleans
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Returns the accuracy value
accuracy = accuracy * 100
# Printing the Accuracy of the Model
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
