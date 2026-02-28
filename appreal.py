import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np


print("Đang tải và xử lý dữ liệu MNIST...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



def build_model():
    model = models.Sequential([
     
        Input(shape=(28, 28, 1)),


        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

     
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  
        layers.Dense(10, activation='softmax')
    ])
    return model


model = build_model()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nĐộ chính xác trên tập kiểm tra: {test_acc * 100:.2f}%')


model.save('mnist_cnn_model.keras')

print("Đã lưu model thành công vào file 'mnist_cnn_model.keras'")
