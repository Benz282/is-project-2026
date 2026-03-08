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

# --- Model Test Page 1 (Ice) ---
elif menu == "Model Test 1 (Ice)":
    st.title("📊 Sensor Value Prediction Model (Ice Skating Data)")
    
    # --- Documentation Section ---
    with st.expander("📖 Model Development Details (Click to expand)", expanded=True):
        st.subheader("1. Data Preparation")
        st.write("""
        - **Dataset:** Compass and sensor measurement data recorded during ice skating.
        - **Data Handling:** Missing values were handled, and data was formatted for statistical processing.
        - **Independent Variable (X):** Timestamp (Recording sequence).
        - **Dependent Variable (y):** Value (The signal value measured by the sensor).
        """)

        st.subheader("2. Algorithm Theory (Ensemble Learning)")
        st.write("""
        This model utilizes a **Voting Regressor** approach, combining results from 3 sub-models to reduce error and improve accuracy:
        1. **Linear Regression:** Analyzes basic linear trends.
        2. **Decision Tree:** Handles non-linear data fluctuations.
        3. **Random Forest:** Uses multiple randomized decision trees to find the most stable average.
        """)

        st.subheader("3. References")
        
        # --- Reference URL Section ---
        st.success("🔗 **Data Source from Kaggle:**")
        st.write("You can check the original dataset here:")
        st.markdown("[Kaggle: Ice Skating Compass Data (Dataset.csv)](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data/data?select=dataset.csv)")
        st.caption("Reference by Frank van Rest (Kaggle Dataset)")

    st.divider()
    
    # --- Brief Info Section (English) ---
    with st.expander("📖 Data Overview (Click to expand)", expanded=True):
        st.subheader("1. Overview")
        st.write("""This dataset focuses on **biomechanical analysis** using **Inertial Measurement Units (IMU)**. It tracks how a skater moves, pushes, and glides on the ice by recording physical data from sensors attached to the skater (likely on the skates or lower limbs). """)

        st.subheader("2. Sensor Data Components")
        st.write("""
        1. **Accelerometer Data (X, Y, Z):** Measures acceleration and G-forces. Helps identify push-off intensity and blade impact.
        2. **Gyroscope Data (Angular Velocity):** Measures foot rotation and orientation. Crucial for understanding stroke transitions.
        3. **Magnetometer Data (Compass):** Provides directional data (Heading) to track the skater's path.
        4. **Timestamps:** High-frequency recording to ensure fluid motion capture.
        """)

        st.subheader("3. Common Use Cases")
        st.write(""" 
        1. **Stroke Identification:** Differentiating between "push-off" and "gliding" phases.
        2. **Performance Metrics:** Calculating stroke frequency, contact time, and speed patterns.
        3. **Machine Learning:** Training algorithms to recognize skating techniques or form inefficiencies.
        """)

        st.write(""" **In short:** It is a technical dataset intended for sports scientists and data analysts to study the physics and efficiency of ice skating through sensor-based motion tracking.
        """)

    # --- Prediction Section ---
    st.subheader("🔮 Prediction Section")
    if model_ice is None:
        st.error("❌ 'model_ice_ensemble.pkl' not found. Please check your file upload.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_val = st.number_input("Enter Timestamp (Example: 1540892278):", value=1540892278.0, format="%.1f")
        
        with col2:
            st.write("") 
            st.write("") 
            if st.button("Start Prediction"):
                prediction = model_ice.predict([[input_val]])
                st.balloons()
                st.success(f"**Predicted Value:** {prediction[0]:.4f}")

        # --- Status Explanation Table ---
        st.divider()
        st.subheader("📋 Movement Status Reference (Example Data)")
        st.write("This table helps analyze skating behavior based on various sensor values:")
        
        # Create Table Data
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





