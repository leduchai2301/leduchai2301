import tensorflow as tf
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt  


print("--- Đang khởi động hệ thống AI ---")
try:
  
    model = tf.keras.models.load_model('mnist_cnn_model.keras')
    print("Mô hình CNN đã sẵn sàng!")
except Exception as e:
    print(f"Lỗi không thể tải mô hình: {e}")



def predict_digit(image):
    if image is None:
        return None


    if isinstance(image, dict):
        image = image['composite']


    image_np = np.array(image)

 
    plt.imsave('raw_sketch.png', image_np)


    img_gray = image_np[:, :, 0]


    img_resized = tf.image.resize(img_gray.reshape(img_gray.shape + (1,)), (28, 28)).numpy()
    img_resized = img_resized.reshape(28, 28) / 255.0


    if np.mean(img_resized) > 0.5:
        img_final = 1.0 - img_resized
    else:
        img_final = img_resized

 
    plt.imsave('processed_input.png', img_final, cmap='gray')

    input_tensor = img_final.reshape(1, 28, 28, 1)

    try:
        prediction = model.predict(input_tensor)
      
        return {str(i): float(prediction[0][i]) for i in range(10)}
    except Exception as e:
        return {f"Lỗi dự đoán: {str(e)}": 0}



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

    submit_btn.click(fn=predict_digit, inputs=input_pad, outputs=output_label)
    clear_btn.click(lambda: None, None, input_pad) 

if __name__ == "__main__":
  

    demo.launch(share=False)
