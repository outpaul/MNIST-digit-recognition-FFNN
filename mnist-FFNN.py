import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x1, y1), (x2, y2) = mnist.load_data()

xtrain = tf.keras.utils.normalize(x1, axis = 1)
xtest = tf.keras.utils.normalize(x2, axis = 1)

model2 = tf.keras.models.Sequential()

model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model2.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model2.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model2.fit(xtrain, y1, epochs = 3)

test_loss, test_accuracy = model2.evaluate(xtest, y2)
print("Test set loss: " + str(test_loss))

l = [np.argmax(i) for i in model2.predict(xtest[5:8])]

plot = plt.figure(figsize = (10,4))
plot.add_subplot(1,3,1)
plt.imshow(x2[5], cmap = plt.cm.binary)
plt.title(l[0])
plot.add_subplot(1,3,2)
plt.imshow(x2[6], cmap = plt.cm.binary)
plt.title(l[1])
plot.add_subplot(1,3,3)
plt.imshow(x2[7], cmap = plt.cm.binary)
plt.title(l[2])
plt.show()
