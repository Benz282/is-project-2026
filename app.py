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
elif menu == "ทดสอบโมเดล 1 (Ice)":
    st.title("📊 การพัฒนาโมเดลทำนายค่าเซนเซอร์ (Ice Skating Data)")
    
    # --- ส่วนอธิบายเนื้อหา (Documentation) ---
    with st.expander("📖 รายละเอียดแนวทางการพัฒนาโมเดล (คลิกเพื่ออ่าน)", expanded=True):
        st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
        st.write("""
        - **Dataset:** ข้อมูลจากการวัดค่าเข็มทิศและเซนเซอร์ (Compass Sensor Data) ขณะเคลื่อนที่บนน้ำแข็ง
        - **การจัดการข้อมูล:** ทำการแทนที่ค่าว่าง (Missing Values) และปรับรูปแบบข้อมูลให้พร้อมสำหรับการประมวลผลทางสถิติ
        - **ตัวแปรต้น (X):** Timestamp (ลำดับเวลาที่บันทึกข้อมูล)
        - **ตัวแปรตาม (y):** Value (ค่าสัญญาณที่วัดได้จากเซนเซอร์)
        """)

        st.subheader("2. ทฤษฎีอัลกอริทึม (Ensemble Learning)")
        st.write("""
        โมเดลนี้ใช้แนวคิด **Voting Regressor** โดยการรวมผลลัพธ์จาก 3 โมเดลย่อย เพื่อลดความผิดพลาด (Error) และเพิ่มความแม่นยำ:
        1. **Linear Regression:** วิเคราะห์แนวโน้มเชิงเส้นพื้นฐาน
        2. **Decision Tree:** จัดการกับความผันผวนของข้อมูลที่ไม่เป็นเส้นตรง
        3. **Random Forest:** ใช้การสุ่มสร้างต้นไม้ตัดสินใจหลายชุดเพื่อหาค่าเฉลี่ยที่เสถียรที่สุด
        """)

        st.subheader("3. แหล่งอ้างอิงข้อมูล (References)")
        
        # --- ส่วนใส่ URL อ้างอิง ---
        st.success("🔗 **แหล่งที่มาข้อมูลจาก Kaggle:**")
        st.write("คุณสามารถตรวจสอบข้อมูลต้นฉบับได้ที่นี่:")
        st.markdown("[Kaggle: Ice Skating Compass Data (Dataset.csv)](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data/data?select=dataset.csv)")
        st.caption("Reference by Frank van Rest (Kaggle Dataset)")

    st.divider()
    # --- ส่วนอธิบายเนื้อหา (Documentation) ---
    
    with st.expander("📖 ข้อมูลโดยสังเขป (คลิกเพื่ออ่าน)", expanded=True):
        st.subheader("1. Overview")
        st.write("""This dataset focuses on **biomechanical analysis** using **Inertial Measurement Units (IMU)**. It tracks how a skater moves, pushes, and glides on the ice by recording physical data from sensors attached to the skater (likely on the skates or lower limbs). """)

        st.subheader("2. The data typically contains time-series data from various sensors")
        st.write("""
        1. **Accelerometer Data (X, Y, Z):** Measures the acceleration and G-forces. This helps identify the intensity of the push-off and the impact of the blade hitting the ice.
        2. **Gyroscope Data (Angular Velocity):** Measures the rotation and orientation of the foot. It is crucial for understanding the transition between strokes and the angle of the skate.
        3. **Magnetometer Data (Compass):** Provides directional data (Heading). This allows researchers to track the skater's path around the rink and during turns.
        4. **Timestamps:** High-frequency recording (often in milliseconds) to ensure the fluid motion of skating is captured accurately.
        """)

        st.subheader("3. Common Use Cases")
        st.write(""" 
        1. **Stroke Identification:** Differentiating between a "push-off" phase and a "gliding" phase.
        2. **Performance Metrics:** Calculating stroke frequency, contact time (how long the blade stays on the ice), and speed patterns.
        3. **Machine Learning:** Training algorithms to automatically recognize skating techniques or detect inefficiencies in a skater's form.
        """)

        st.write(""" **In short:** It is a technical dataset intended for sports scientists and data analysts to study the physics and efficiency of ice skating through sensor-based motion tracking.
        """)

    # --- ส่วนการทำนายผล ---
    st.subheader("🔮 ส่วนการทดสอบทำนายผล (Prediction)")
    if model_ice is None:
        st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl กรุณาตรวจสอบการอัปโหลดไฟล์")
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

    # --- เพิ่มตารางแสดงสถานะด้านล่างการทำนาย ---
        st.divider()
        st.subheader("📋 ตารางอธิบายสถานะการเคลื่อนที่ (ตัวอย่างข้อมูล)")
        st.write("ตารางนี้ช่วยวิเคราะห์พฤติกรรมการเล่นสเก็ตน้ำแข็งจากค่าเซนเซอร์ต่างๆ:")
        
        # สร้างข้อมูลตาราง
        data_info = {
            "Time (ms)": [1000, 1100, 1200, 1300],
            "Accel_X (Push)": [0.2, 2.5, 0.5, 0.1],
            "Gyro_Z (Rotation)": [5.1, 12.4, 45.0, 2.0],
            "Compass (Heading)": ["180°", "182°", "210°", "230°"],
            "Status (Description)": [
                "Gliding: Moving forward smoothly, low impact.",
                "Push-off: Executing a push against the ice (High Accel).",
                "Turning: Cornering (Rapid change in Gyro and Heading).",
                "Gliding: Returning to a steady glide."
            ]
        }
        
        df_info = pd.DataFrame(data_info)
        
        # Display Table
        st.table(df_info)
        
        st.caption("🔍 Note: Accel and Gyro values are simulated to illustrate movement characteristics over time (Timestamp).")
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









