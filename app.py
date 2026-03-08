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
    st.title("📊 Sensor Analytics & Predictive Modeling")
    st.caption("Project: Ice Skating Compass Data Analysis | Model: Ensemble Voting Regressor")

    # ใช้ Tabs เพื่อความเป็นสากลและอ่านง่าย
    tab1, tab2, tab3 = st.tabs(["📑 Documentation", "🔮 Model Testing", "📈 Data Insight"])

    with tab1:
        st.subheader("1. แนวทางการพัฒนาโมเดล (Data Preparation)")
        st.write("""
        - **Dataset:** ข้อมูลจากการวัดค่าเข็มทิศและเซนเซอร์ (Compass Sensor Data) ขณะเคลื่อนที่บนน้ำแข็ง
        - **Algorithm:** โมเดลนี้ใช้แนวคิด **Voting Regressor** โดยการรวมผลลัพธ์จาก Linear Regression, Decision Tree และ Random Forest
        """)

        st.subheader("2. Overview & Data Structure")
        st.write("""
        This dataset focuses on **biomechanical analysis** using **Inertial Measurement Units (IMU)**. 
        It tracks how a skater moves, pushes, and glides on the ice by recording physical data:
        1. **Accelerometer Data (X, Y, Z):** Measures acceleration and G-forces (Push-off intensity).
        2. **Gyroscope Data (Angular Velocity):** Measures rotation and orientation of the foot.
        3. **Magnetometer Data (Compass):** Provides directional data (Heading).
        4. **Timestamps:** High-frequency recording to ensure fluid motion is captured accurately.
        """)

        st.subheader("3. แหล่งอ้างอิงข้อมูล (References)")
        st.success("🔗 **Source:** [Kaggle: Ice Skating Compass Data](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data/data?select=dataset.csv)")
        st.caption("Reference by Frank van Rest (Kaggle Dataset)")

    with tab2:
        st.subheader("🔮 Predictive Analytics")
        if model_ice is None:
            st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl")
        else:
            container = st.container(border=True)
            with container:
                input_val = st.number_input("ป้อนค่า Timestamp (ตัวอย่าง: 1540892278):", value=1540892278.0, format="%.1f")
                btn_predict = st.button("เริ่มการทำนายผล", use_container_width=True)

            if btn_predict:
                prediction = model_ice.predict([[input_val]])
                st.balloons()
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Predicted Value", f"{prediction[0]:.4f}")
                col_m2.metric("Status", "Calculated")

    with tab3:
        st.subheader("📋 ตารางอธิบายสถานะการเคลื่อนที่")
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
        st.table(pd.DataFrame(data_info))
        st.caption("🔍 Note: Values are simulated for educational movement analysis.")

# --- หน้าทดสอบโมเดล 2 (MNIST) ---
elif menu == "ทดสอบโมเดล 2 (MNIST)":
    st.title("🧠 Neural Network (MNIST Classifier)")
    st.write("⚠️ อัปโหลดรูปภาพตัวเลขเพียงตัวเดียว (ขาวบนดำ)")
    
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
                st.error(f"⚠️ ตรวจพบตัวเลข {digit_count} ตัว! กรุณาใช้รูปตัวเลขเดียว")
            elif digit_count == 0:
                st.warning("❓ ไม่พบตัวเลข")
            else:
                if st.button("วิเคราะห์ตัวเลข"):
                    img_resized = image.resize((28, 28))
                    img_array = np.array(img_resized)
                    img_array = np.where(img_array > 100, 255, 0) # ปรับความชัด
                    img_input = img_array.astype('float32') / 255.0
                    img_input = img_input.reshape(1, 28, 28, 1)
                    
                    res = model_mnist.predict(img_input)
                    st.success(f"🎯 AI วิเคราะห์ว่าเป็นเลข: {np.argmax(res)}")
                    st.write(f"ความเชื่อมั่น: {np.max(res)*100:.2f}%")
