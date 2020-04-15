import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

print(test_labels[:8])

# model = keras.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(3000, input_shape=(28, 28),  activation='relu'),
#     keras.layers.Dense(1000, activation="relu"),
#     keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=5)

model = keras.models.load_model("new_model_1")

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy: {}".format(test_acc))

predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
print(class_names[np.argmax(predictions[1])])
print(class_names[np.argmax(predictions[2])])
print(class_names[np.argmax(predictions[3])])


#model.save("new_model_1")
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
plt.imshow(test_images[1], cmap=plt.cm.binary)
plt.show()
plt.imshow(test_images[2], cmap=plt.cm.binary)
plt.show()
plt.imshow(test_images[3], cmap=plt.cm.binary)

plt.show()