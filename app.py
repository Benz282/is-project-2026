import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2  # ใช้สำหรับนับจำนวนตัวเลข

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Project IS 2568", layout="wide")

# --- ฟังก์ชันโหลดโมเดล (โหลดครั้งเดียวเพื่อความเร็ว) ---
@st.cache_resource
def load_all_models():
    try:
        m1 = joblib.load('model_ice_ensemble.pkl')
        m2 = tf.keras.models.load_model('model_mnist_nn.h5')
        return m1, m2
    except Exception as e:
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
elif menu == "ทดสอบmoเดล 1 (Ice)":
    st.title("📊 Sensor Analytics & Predictive Modeling")
    st.caption("Project: Ice Skating Compass Data Analysis | Model: Ensemble Voting Regressor")

    # --- ส่วนที่ 1: การจัดแบ่งเนื้อหาด้วย Tabs (แบบสากล) ---
    tab1, tab2, tab3 = st.tabs(["📑 Documentation", "🔮 Model Testing", "📈 Data Insight"])

    # --- Tab 1: รายละเอียดวิชาการและที่มาของข้อมูล ---
    with tab1:
        st.subheader("Dataset Overview")
        st.write("""
        The **"Ice Skating Compass Data"** dataset (specifically the `dataset.csv` file) provided by **Frank van Rest on Kaggle** is a collection of sensor data used to analyze the movement of ice skaters. It focuses on biomechanical analysis 
        using **Inertial Measurement Units (IMU)** to track how a skater moves, pushes, and glides.
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **1. Data Structure (dataset.csv)**
            - **Accelerometer (X, Y, Z):** Measures G-forces to identify push-off intensity and blade impact.
            - **Gyroscope (Angular Velocity):** Measures rotation/orientation to understand stroke transitions.
            - **Magnetometer (Compass):** Provides directional data (Heading) to track the skater's path.
            - **Timestamps:** High-frequency recording (ms) to capture fluid motion accurately.
            """)
        with col_b:
            st.markdown("""
            **2. Common Use Cases**
            - **Stroke Identification:** Differentiating between "push-off" and "gliding" phases.
            - **Performance Metrics:** Calculating stroke frequency, contact time, and speed patterns.
            - **Machine Learning:** Training algorithms to recognize skating techniques or detect inefficiencies.
            """)
        
        st.divider()
        st.subheader("Algorithm Theory")
        st.write("""
        โมเดลนี้ใช้แนวคิด **Voting Regressor (Ensemble Learning)** โดยการรวมผลลัพธ์จาก 3 โมเดลย่อย เพื่อลด Error และเพิ่มความแม่นยำ:
        - **Linear Regression:** วิเคราะห์แนวโน้มเชิงเส้นพื้นฐาน
        - **Decision Tree:** จัดการกับความผันผวนของข้อมูลที่ไม่เป็นเส้นตรง
        - **Random Forest:** ใช้การสุ่มสร้างต้นไม้ตัดสินใจหลายชุดเพื่อหาค่าเฉลี่ยที่เสถียรที่สุด
        """)

        st.success("🔗 **Reference & Source:**")
        st.markdown("[Kaggle: Ice Skating Compass Data (Dataset.csv)](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data/data?select=dataset.csv)")
        st.caption("Reference by Frank van Rest (Kaggle Dataset)")

    # --- Tab 2: ส่วนการทดสอบทำนายผล ---
    with tab2:
        st.subheader("Predictive Analytics")
        st.write("ป้อนค่า Timestamp เพื่อจำลองการทำนายค่าสัญญาณเซนเซอร์ (Predictive Sensor Value)")
        
        container = st.container(border=True)
        with container:
            input_val = st.number_input("Input Timestamp:", value=1540892278.0, format="%.1f")
            btn_predict = st.button("Run Prediction", use_container_width=True)

        if btn_predict:
            if model_ice is not None:
                prediction = model_ice.predict([[input_val]])
                st.write("---")
                m1, m2 = st.columns(2)
                m1.metric(label="Predicted Value", value=f"{prediction[0]:.4f}")
                m2.metric(label="Status", value="Success", delta="Stable")
                st.success("การวิเคราะห์เสร็จสมบูรณ์: โมเดลคำนวณผลลัพธ์ตามรูปแบบข้อมูลย้อนหลังเรียบร้อยแล้ว")
                st.balloons()
            else:
                st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบการอัปโหลดไฟล์ .pkl ขึ้น GitHub")

    # --- Tab 3: ตารางข้อมูลและสถานะการเคลื่อนที่ ---
    with tab3:
        st.subheader("Movement Phase Reference Matrix")
        st.write("ตารางอธิบายความสัมพันธ์ระหว่างค่าเซนเซอร์และพฤติกรรมการเคลื่อนที่บนน้ำแข็ง")
        
        df_status = pd.DataFrame({
            "Time (ms)": [1000, 1100, 1200, 1300],
            "Accel_X (แรงถีบ)": [0.2, 2.5, 0.5, 0.1],
            "Gyro_Z (การหมุน)": [5.1, 12.4, 45.0, 2.0],
            "Compass (ทิศทาง)": ["180°", "182°", "210°", "230°"],
            "สถานะ (คำอธิบาย)": [
                "Gliding: ลื่นไถลไปข้างหน้า แรงกระแทกต่ำ",
                "Push-off: เริ่มออกแรงถีบ (Accel พุ่งสูง)",
                "Turning: กำลังเข้าโค้ง (Gyro เปลี่ยนเร็ว)",
                "Gliding: กลับมาลื่นไถลนิ่งๆ อีกครั้ง"
            ]
        })
        
        st.dataframe(df_status, use_container_width=True, hide_index=True)
        st.caption("Note: This table represents idealized sensor patterns for educational purposes.")
# --- หน้าทดสอบโมเดล 2 (MNIST) ---
elif menu == "ทดสอบโมเดล 2 (MNIST)":
    st.header("🧠 ทดสอบโมเดล Neural Network (จำแนกตัวเลข)")
    st.write("⚠️ เงื่อนไข: อัปโหลดรูปภาพที่มี **ตัวเลขเพียงตัวเดียว** (ตัวเลขสีขาว พื้นหลังสีดำ)")
    
    if model_mnist is None:
        st.error("❌ ไม่พบไฟล์ model_mnist_nn.h5 ในระบบ")
    else:
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # 1. แปลงไฟล์ภาพและแสดงผล
            image = Image.open(uploaded_file).convert('L')
            st.image(image, caption='รูปที่อัปโหลด', width=150)
            
            # 2. ตรวจสอบจำนวนตัวเลขด้วย OpenCV
            img_cv = np.array(image)
            _, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # กรองเอาเฉพาะกลุ่มพิกเซลที่มีขนาดใหญ่พอ (ป้องกัน Noise)
            digit_count = sum(1 for c in contours if cv2.contourArea(c) > 30)

            # 3. ตรวจสอบเงื่อนไขจำนวนตัวเลข
            if digit_count > 1:
                st.error(f"⚠️ ตรวจพบตัวเลข {digit_count} ตัว! กรุณาใช้รูปที่มีตัวเลขเพียงตัวเดียว")
            elif digit_count == 0:
                st.warning("❓ ไม่พบตัวเลขในรูปภาพ")
            else:
                # ถ้าผ่านเงื่อนไข ให้แสดงปุ่มวิเคราะห์
                if st.button("วิเคราะห์ตัวเลข"):
                    # เตรียมรูป (Pre-processing)
                    img_resized = image.resize((28, 28))
                    img_array = np.array(img_resized)
                    
                    # ปรับความชัด (Threshold) ให้เหมือนตอนเทรน
                    img_array = np.where(img_array > 100, 255, 0)
                    
                    img_input = img_array.astype('float32') / 255.0
                    img_input = img_input.reshape(1, 28, 28, 1)
                    
                    # ทำนาย
                    res = model_mnist.predict(img_input)
                    final_res = np.argmax(res)
                    confidence = np.max(res) * 100
                    
                    st.success(f"🎯 AI วิเคราะห์ว่าเป็นเลข: {final_res}")
                    st.write(f"ความเชื่อมั่น: {confidence:.2f}%")







