import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np

# 1. Chuẩn bị dữ liệu (Data Preparation)
print("Đang tải và xử lý dữ liệu MNIST...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu về khoảng [0, 1] (Normalization)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Thêm chiều kênh màu (Channel dimension) vì CNN yêu cầu input dạng (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# 2. Xây dựng mô hình (Model Architecture) - Đã sửa theo chuẩn Keras mới nhất
def build_model():
    model = models.Sequential([
        # Sử dụng lớp Input riêng biệt để tránh cảnh báo (Warning)
        Input(shape=(28, 28, 1)),

        # Trích xuất đặc trưng
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Phân loại
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Chống học vẹt
        layers.Dense(10, activation='softmax')
    ])
    return model


model = build_model()

# 3. Cấu hình huấn luyện (Compile)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Bắt đầu huấn luyện (Training)
print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
# Epochs = 5 là đủ để thấy kết quả tốt (>98%)
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Đánh giá và Lưu model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nĐộ chính xác trên tập kiểm tra: {test_acc * 100:.2f}%')

# Lưu model để dùng lại sau này (quan trọng cho bước làm App)
model.save('mnist_cnn_model.keras')
print("Đã lưu model thành công vào file 'mnist_cnn_model.keras'")