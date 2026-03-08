import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 # สำหรับตรวจจับวัตถุในภาพ

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# --- โหลดโมเดล ---
@st.cache_resource
def load_mnist_model():
    return tf.keras.models.load_model('model_mnist_nn.h5')

try:
    model = load_mnist_model()
except:
    st.error("ไม่พบไฟล์ model_mnist_nn.h5 กรุณาอัปโหลดไฟล์โมเดลขึ้น GitHub")

st.title("🧠 เครื่องมือวิเคราะห์ตัวเลขเขียนมือ")
st.write("อัปโหลดรูปภาพตัวเลข (0-9) เพื่อให้ AI ทำนาย")

# --- ส่วนอัปโหลดไฟล์ ---
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (พื้นหลังสีดำ ตัวเลขสีขาว)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. แสดงรูปต้นฉบับ
    raw_image = Image.open(uploaded_file).convert('L') # แปลงเป็น Grayscale
    st.image(raw_image, caption="รูปภาพที่คุณอัปโหลด", width=200)

    # 2. ระบบคัดกรองจำนวนตัวเลขด้วย OpenCV (Contours)
    img_cv = np.array(raw_image)
    # ทำ Threshold เพื่อให้ภาพชัดเจน (ขาว-ดำ สนิท)
    _, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
    # ค้นหาเส้นรอบรูป (กลุ่มก้อนวัตถุ)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # กรองเฉพาะกลุ่มก้อนที่มีขนาดใหญ่พอจะเป็นตัวเลข (ป้องกันจุดรบกวน)
    valid_digits = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
    digit_count = len(valid_digits)

    # 3. เงื่อนไขการแจ้งเตือน
    if digit_count > 1:
        st.error(f"⚠️ ตรวจพบตัวเลข {digit_count} ตัวในภาพ! ระบบรองรับการทำนายเพียงครั้งละ 1 ตัวเลขเท่านั้น")
    elif digit_count == 0:
        st.warning("❌ ไม่พบตัวเลขในรูปภาพ กรุณาลองใหม่อีกครั้ง")
    else:
        # กรณีพบ 1 ตัวเลขถ้วน -> ทำการทำนาย
        if st.button("วิเคราะห์ผลลัพธ์"):
            # เตรียมภาพให้พร้อมสำหรับ Model (28x28)
            img_resized = raw_image.resize((28, 28))
            img_input = np.array(img_resized) / 255.0 # Normalization
            img_input = img_input.reshape(1, 28, 28, 1) # Reshape ให้ตรงตาม Model Input

            # ส่งให้ AI ทาย
            prediction = model.predict(img_input)
            result = np.argmax(prediction)
            
            st.success(f"🎯 AI วิเคราะห์ว่าเป็นตัวเลข: {result}")
            st.balloons()
