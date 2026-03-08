import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Project IS 2568", layout="wide")

# --- ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_all_models():
    try:
        m1 = joblib.load('model_ice_ensemble.pkl')
        m2 = tf.keras.models.load_model('model_mnist_nn.h5')
        return m1, m2
    except:
        return None, None

model_ice, model_mnist = load_all_models()

# --- เมนูแถบข้าง (Sidebar) ---
st.sidebar.title("เมนูควบคุม")
menu = st.sidebar.selectbox("เลือกหน้า", ["หน้าแรก", "ทดสอบโมเดล 1 (Ice)", "ทดสอบโมเดล 2 (MNIST)"])

# --- หน้าแรก ---
if menu == "หน้าแรก":
    st.title("🤖 โปรเจค AI - วิชา IS 2568")
    st.write("ยินดีต้อนรับสู่โปรเจคการพัฒนา Machine Learning และ Neural Network")
    st.markdown("""
    ### รายละเอียดโปรเจค:
    - **โมเดล 1:** ทำนายค่าเซนเซอร์ (Ice Dataset) โดยใช้ Ensemble Learning
    - **โมเดล 2:** จำแนกตัวเลขเขียนมือ (MNIST) โดยใช้ Convolutional Neural Network (CNN)
    """)
    st.info("💡 กรุณาเลือกเมนูด้านซ้ายเพื่อเริ่มทดสอบ")

# --- หน้าทดสอบโมเดล 1 (Ice) ---
elif menu == "ทดสอบโมเดล 1 (Ice)":
    st.title("📊 การพัฒนาโมเดลทำนายค่าเซนเซอร์ (Ice Skating Data)")
    
    # 1. ส่วนอธิบายแนวทาง (แบบเก่า: Expander)
    with st.expander("📖 รายละเอียดแนวทางการพัฒนาโมเดล (คลิกเพื่ออ่าน)", expanded=True):
        st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
        st.write("""
        - **Dataset:** ข้อมูลจากการวัดค่าเข็มทิศและเซนเซอร์ (Compass Sensor Data) ขณะเคลื่อนที่บนน้ำแข็ง
        - **การจัดการข้อมูล:** ทำการแทนที่ค่าว่าง (Missing Values) และปรับรูปแบบข้อมูลให้พร้อมสำหรับการประมวลผล
        - **ตัวแปรต้น (X):** Timestamp | **ตัวแปรตาม (y):** Value
        """)

        st.subheader("2. ทฤษฎีอัลกอริทึม (Ensemble Learning)")
        st.write("""
        โมเดลนี้ใช้แนวคิด **Voting Regressor** โดยการรวมผลลัพธ์จาก 3 โมเดลย่อย:
        1. Linear Regression | 2. Decision Tree | 3. Random Forest
        """)

        st.subheader("3. แหล่งอ้างอิงข้อมูล (References)")
        st.success("🔗 **Source:** [Kaggle: Ice Skating Compass Data](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data/data?select=dataset.csv)")
        st.caption("Reference by Frank van Rest (Kaggle Dataset)")

    # 2. ส่วนข้อมูลภาษาอังกฤษ (แบบเก่า: Expander)
    with st.expander("📖 Kaggle Summary", expanded=False):
        st.subheader("Overview")
        st.write("This dataset focuses on biomechanical analysis using **Inertial Measurement Units (IMU)**. It tracks how a skater moves, pushes, and glides.")
        st.subheader("Data Structure")
        st.write("""
        - **Accelerometer:** Measures G-forces and push-off intensity.
        - **Gyroscope:** Measures rotation and foot orientation.
        - **Magnetometer:** Provides directional heading data.
        - **Timestamps:** High-frequency recording in milliseconds.
        """)
        st.subheader("Common Use Cases")
        st.write("Stroke Identification, Performance Metrics, and Machine Learning technique detection.")

    st.divider()

    # 3. ส่วนการทำนายผล
    st.subheader("🔮 ส่วนการทดสอบทำนายผล (Prediction)")
    if model_ice is None:
        st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_val = st.number_input("ป้อนค่า Timestamp (ตัวอย่าง: 1540892278):", value=1540892278.0, format="%.1f")
        with col2:
            st.write("")
            st.write("")
            if st.button("เริ่มการทำนายผล"):
                prediction = model_ice.predict([[input_val]])
                st.balloons()
                st.success(f"**ค่า Value ที่ทำนายได้คือ:** {prediction[0]:.4f}")

    # 4. ตารางสถานะ (เรียงต่อด้านล่าง)
    st.divider()
    st.subheader("📋 ตารางอธิบายสถานะการเคลื่อนที่")
    data_info = {
        "Time (ms)": [1000, 1100, 1200, 1300],
        "Accel_X (Push)": [0.2, 2.5, 0.5, 0.1],
        "Gyro_Z (Rotation)": [5.1, 12.4, 45.0, 2.0],
        "Compass (Heading)": ["180°", "182°", "210°", "230°"],
        "Status (Description)": [
            "Gliding: ลื่นไถลไปข้างหน้า แรงกระแทกต่ำ",
            "Push-off: เริ่มออกแรงถีบ (High Accel)",
            "Turning: กำลังเข้าโค้ง (Gyro เปลี่ยนเร็ว)",
            "Gliding: กลับมาลื่นไถลนิ่งๆ อีกครั้ง"
        ]
    }
    st.table(pd.DataFrame(data_info))

# --- หน้าทดสอบโมเดล 2 (MNIST) ---
elif menu == "ทดสอบโมเดล 2 (MNIST)":
    st.header("🧠 ทดสอบโมเดล Neural Network (MNIST)")
    if model_mnist is None:
        st.error("❌ ไม่พบไฟล์ model_mnist_nn.h5")
    else:
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            img_cv = np.array(image)
            _, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_count = sum(1 for c in contours if cv2.contourArea(c) > 30)
            
            st.image(image, caption='รูปที่อัปโหลด', width=150)
            if digit_count > 1:
                st.error(f"⚠️ ตรวจพบตัวเลข {digit_count} ตัว! กรุณาใช้รูปเดียว")
            elif digit_count == 0:
                st.warning("❓ ไม่พบตัวเลข")
            else:
                if st.button("วิเคราะห์ตัวเลข"):
                    img_resized = image.resize((28, 28))
                    img_array = np.array(img_resized)
                    img_array = np.where(img_array > 100, 255, 0)
                    img_input = img_array.astype('float32') / 255.0
                    img_input = img_input.reshape(1, 28, 28, 1)
                    res = model_mnist.predict(img_input)
                    st.success(f"🎯 AI วิเคราะห์ว่าเป็นเลข: {np.argmax(res)}")

