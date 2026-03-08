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
    
    # 1. ส่วนอธิบายแนวทาง (แบบเก่า: Expander)
    with st.expander("📖 Model Development Details (Click to read)", expanded=True):
        st.subheader("1. Data Preparation")
        st.write("""
        - **Dataset:** Compass and sensor measurement data recorded during ice skating movements.
        - **Data Management:** Handled missing values and preprocessed the data format for statistical analysis.
        - **Independent Variable (X):** Timestamp | **Dependent Variable (y):** Value
        """)

        st.subheader("2. Algorithm Theory (Ensemble Learning)")
        st.write("""
        This model utilizes the **Voting Regressor** concept by ensemble-averaging the predictions from 3 base models to minimize error and maximize accuracy:
        1. **Linear Regression:** Captures the basic linear relationship.
        2. **Decision Tree:** Manages non-linear fluctuations in the data.
        3. **Random Forest:** Aggregates multiple decision trees for a more stable and robust prediction.
        """)

        st.subheader("3. Data References")
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
    st.subheader("🔮 Model Prediction Testing")
    if model_ice is None:
        st.error("❌ Not found model_ice_ensemble.pkl")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_val = st.number_input("Enter Timestamp (Example: 1540892278):", value=1540892278.0, format="%.1f")
        with col2:
            st.write("")
            st.write("")
            if st.button("Run Prediction"):
                prediction = model_ice.predict([[input_val]])
                st.snow()
                st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6; border: 1px solid #d1d5db;">
                <p style="margin: 0; font-size: 16px; color: #31333F;">Predicted Sensor Value</p>
                <h2 style="margin: 0; color: #007bff; font-size: 36px;">{prediction[0]:.4f}</h2>
            </div>
        """, unsafe_allow_html=True)

    # 4. ตารางสถานะ (เรียงต่อด้านล่าง)
    st.divider()
    st.subheader("📋 Movement Status Reference Table")
    st.caption("This table assists in analyzing ice skating behavior based on various sensor metrics:")
    data_info = {
        "Time (ms)": [1000, 1100, 1200, 1300],
        "Accel_X (Push)": [0.2, 2.5, 0.5, 0.1],
        "Gyro_Z (Rotation)": [5.1, 12.4, 45.0, 2.0],
        "Compass (Heading)": ["180°", "182°", "210°", "230°"],
        "Status (Description)": [
            "Gliding: Steady forward motion with low impact.",
            "Push-off: Executing a stroke against the ice (Spike in Acceleration).",
            "Turning: Cornering or changing direction (Rapid Gyro/Heading change).",
            "Gliding: Returning to a steady glide."
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




