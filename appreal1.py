import tensorflow as tf
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt  # Dùng để xuất ảnh minh họa cho báo cáo

# 1. Tải model đã huấn luyện
print("--- Đang khởi động hệ thống AI ---")
try:
    # Đảm bảo file 'mnist_cnn_model.keras' nằm cùng thư mục với file code này
    model = tf.keras.models.load_model('mnist_cnn_model.keras')
    print("Mô hình CNN đã sẵn sàng!")
except Exception as e:
    print(f"Lỗi không thể tải mô hình: {e}")


# 2. Hàm xử lý ảnh và dự đoán
def predict_digit(image):
    if image is None:
        return None

    # --- BƯỚC 1: Trích xuất dữ liệu từ Gradio ---
    # Gradio Sketchpad trả về một dictionary, chúng ta lấy ảnh tổng hợp (composite)
    if isinstance(image, dict):
        image = image['composite']

    # Chuyển dữ liệu sang mảng Numpy để xử lý
    image_np = np.array(image)

    # LƯU ẢNH GỐC ĐỂ LÀM BÁO CÁO (Ảnh nền trắng nét đen)
    plt.imsave('raw_sketch.png', image_np)

    # --- BƯỚC 2: Tiền xử lý (Preprocessing) ---
    # Chuyển về ảnh xám bằng cách lấy kênh màu đầu tiên (R)
    # Ảnh ban đầu thường là (H, W, 4) do có kênh Alpha
    img_gray = image_np[:, :, 0]

    # Resize về kích thước 28x28 theo chuẩn MNIST
    img_resized = tf.image.resize(img_gray.reshape(img_gray.shape + (1,)), (28, 28)).numpy()
    img_resized = img_resized.reshape(28, 28) / 255.0

    # --- BƯỚC 3: Thuật toán Đảo ngược màu (Invert Colors) ---
    # Mục tiêu: Biến Nền trắng/Chữ đen thành Nền đen/Chữ trắng (Chuẩn MNIST)
    if np.mean(img_resized) > 0.5:
        img_final = 1.0 - img_resized
    else:
        img_final = img_resized

    # LƯU ẢNH SAU XỬ LÝ ĐỂ LÀM BÁO CÁO (Ảnh 28x28, nền đen nét trắng)
    plt.imsave('processed_input.png', img_final, cmap='gray')

    # --- BƯỚC 4: Dự đoán với mô hình CNN ---
    # Định dạng lại tensor đầu vào: (1 batch, 28 height, 28 width, 1 channel)
    input_tensor = img_final.reshape(1, 28, 28, 1)

    try:
        prediction = model.predict(input_tensor)
        # Lấy top 10 xác suất cho các chữ số từ 0-9
        return {str(i): float(prediction[0][i]) for i in range(10)}
    except Exception as e:
        return {f"Lỗi dự đoán: {str(e)}": 0}


# 3. Thiết lập giao diện người dùng (Gradio UI)
with gr.Blocks(title="AI Handwritten Digit Recognition") as demo:
    gr.Markdown("# 🖋️ Hệ thống Nhận dạng Chữ số Viết tay")
    gr.Markdown("Vẽ một chữ số bất kỳ vào khung bên dưới để AI phân tích.")

    with gr.Row():
        with gr.Column():
            input_pad = gr.Sketchpad(label="Bảng vẽ (Vẽ số 0-9)", type="numpy")
            submit_btn = gr.Button("Dự đoán", variant="primary")
            clear_btn = gr.Button("Xóa")

        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Kết quả phân tích")

    # Thiết lập sự kiện
    submit_btn.click(fn=predict_digit, inputs=input_pad, outputs=output_label)
    clear_btn.click(lambda: None, None, input_pad)  # Xóa bảng vẽ

# 4. Chạy ứng dụng
if __name__ == "__main__":
    # share=True sẽ tạo link công khai nếu bạn muốn gửi cho bạn bè/giảng viên xem từ xa
    demo.launch(share=False)