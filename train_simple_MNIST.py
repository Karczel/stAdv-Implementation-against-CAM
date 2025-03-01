import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from load_datasets import *  # Assuming you have a separate file to load the dataset

# !THIS DIDN'T WORK, USE THE MNIST FILES IN preset_models!

# Load MNIST data
x_train, y_train, x_test, y_test = load_mnist()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create placeholders for input data and labels
input_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
input_labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

# Build the model manually
conv1 = tf.layers.conv2d(input_images, 64, (5, 5), activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), 2)
conv2 = tf.layers.conv2d(pool1, 64, (5, 5), activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), 2)
flatten = tf.layers.flatten(pool2)
dense1 = tf.layers.dense(flatten, 64, activation=tf.nn.relu)
logits = tf.layers.dense(dense1, 10)

# Loss function and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

# Accuracy metric
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create a session
sess = tf.compat.v1.Session()

# Initialize all variables
sess.run(tf.compat.v1.global_variables_initializer())

# Define the saver
saver = tf.compat.v1.train.Saver()

# Training loop
epochs = 5
batch_size = 64


def get_batches(x_data, y_data, batch_size):
    """
    Generator function to return mini-batches from x_data and y_data.
    """
    # Shuffle the data
    indices = np.arange(x_data.shape[0])  # Get indices of the data
    np.random.shuffle(indices)  # Shuffle indices to randomize the data

    # Split data into batches
    for i in range(0, len(x_data), batch_size):
        batch_indices = indices[i:i + batch_size]  # Select indices for this batch
        yield x_data[batch_indices], y_data[batch_indices]  # Return the batch


for epoch in range(epochs):
    # Example loop for batching
    # You should implement batching logic to split the data into batches
    for batch_x, batch_y in get_batches(x_train, y_train, batch_size):  # Assuming you have a batching function
        # Run the optimizer (training step)
        sess.run(optimizer, feed_dict={input_images: batch_x, input_labels: batch_y})

    # Evaluate accuracy on the test set after every epoch
    test_acc = sess.run(accuracy, feed_dict={input_images: x_test, input_labels: y_test})
    print(f"Epoch {epoch + 1}, Test Accuracy: {test_acc}")

# Save the trained model
save_path = os.path.join('saved_models', 'simple_mnist')
saver.save(sess, save_path)
print(f"Model saved in path: {save_path}")