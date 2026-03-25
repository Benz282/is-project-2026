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
    
    # 1. ส่วนอธิบายแนวทาง
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

    st.divider()

    # --- ส่วนการทำนายผลแบบจัดวางตรงกลาง ---
    st.subheader("🔮 Model Prediction Testing")

    # สร้าง 3 คอลัมน์ เพื่อจัดให้อยู่ตรงกลาง (สัดส่วน 1:2:1)
    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        if model_ice is None:
            st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl")
        else:
            # 1. ส่วนรับข้อมูล
            input_val = st.number_input(
                "Enter Timestamp (Example: 1540892278):", 
                value=1540892278.0, 
                format="%.1f"
            )
            
           # 2. ปุ่มกด
    if st.button("Run Prediction", use_container_width=True):
        # สร้างตัวแปร prediction ขึ้นมาจากการกดปุ่ม
        prediction = model_ice.predict([[input_val]])
        st.snow()
        
        
        # 3. ต้องย่อหน้าให้ st.markdown อยู่ข้างใน if เท่านั้น (สำคัญมาก!)
        st.markdown(f"""
            <div style="text-align: center; padding: 25px; border-radius: 15px; background-color: #e8f5e9; border: 2px solid #c8e6c9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 14px; color: #2e7d32; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">Predicted Sensor Value</p>
                <h1 style="margin: 10px 0; color: #1b5e20; font-size: 48px;">{prediction[0]:.4f}</h1>
                <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 15px 0;">
                <p style="margin: 0; font-size: 16px; color: #2e7d32; font-weight: bold;">
                </p>
                <div style="background-color: #ffffff; border-radius: 10px; height: 8px; margin-top: 10px;">
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.caption("<center style='margin-top:10px;'>Confidence score is based on Model performance during validation</center>", unsafe_allow_html=True)

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
    st.title("📊 Sensor Analytics & Predictive Modeling")
    st.caption("Project: Ice Skating Compass Data Analysis | Model: Ensemble Voting Regressor")
    
    # 1. ส่วนอธิบายแนวทาง
    with st.expander("📖 Model Development Details (Click to read)", expanded=True):
        st.subheader("1. Data Preparation")
        st.write("""
        While both approaches start with the same raw data, the preprocessing steps differ based on the algorithm's architecture:
        - **Dataset:** Handwritten digits (0-9) represented as grayscale images of 28x28 pixels.
        - **Data Management:** - **Normalization:** Scaled pixel values from [0, 255] to [0, 1] to improve convergence speed.
            - **Flattening:** Reshaped 2D images (28x28) into 1D vectors (784 features) for compatibility with ML algorithms.
        - **Independent Variable (X):** 784 Pixel Intensity Values | **Dependent Variable (y):** Digit Label (0-9).
        """)

        st.subheader("2. Algorithm Theory (Neural Network)")
        st.write("""
        This model utilizes the **Voting Classifier** concept, combining multiple 'weak learners' to create a 'strong learner' to maximize classification accuracy:
        1. **Logistic Regression:** Serves as a baseline linear classifier to identify simple pixel-to-digit correlations.
        2. **Random Forest (Bagging):** An ensemble of multiple Decision Trees that reduces variance and prevents overfitting by averaging predictions.
        3. **Support Vector Machine (SVM):** Finds the optimal hyperplane to separate digit classes in a high-dimensional space.

        **Concept:** The final prediction is determined by **Soft Voting**, where the model averages the probability scores from all three base models.
        """)

        st.subheader("3. Data References")
        st.success("🔗 **Source:** [Kaggle: Digit Recognizer (MNIST)](https://www.kaggle.com/c/digit-recognizer/data?select=sample_submission.csv)")
        st.caption("Reference by Kaggle Competition: Digit Recognizer Dataset")

    st.divider()

    # --- ส่วนการทำนายผลแบบจัดวางตรงกลาง ---
    st.subheader("🔮 Model Prediction Testing")

    if model_mnist is None:
        st.error("❌ Don't found model_mnist_nn.h5")
    else:
        uploaded_file = st.file_uploader("Select picture...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            img_cv = np.array(image)
            _, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_count = sum(1 for c in contours if cv2.contourArea(c) > 30)
            
            st.image(image, caption='รูปที่อัปโหลด', width=150)
            if digit_count > 1:
                st.error(f"⚠️ {digit_count} numbers found! Please use a single image.")
            elif digit_count == 0:
                st.warning("❓ Could not find any numbers.")
            else:
                if st.button("Analyze numbers", use_container_width=True):
                    img_resized = image.resize((28, 28))
                    img_array = np.array(img_resized)
                    img_array = np.where(img_array > 100, 255, 0)
                    img_input = img_array.astype('float32') / 255.0
                    img_input = img_input.reshape(1, 28, 28, 1)
                    res = model_mnist.predict(img_input)
                    st.success(f"🎯 AI Detected as number: {np.argmax(res)}")
                    st.digit()
            
           # 2. ปุ่มกด
    if st.button("Run Prediction", use_container_width=True):
        # สร้างตัวแปร prediction ขึ้นมาจากการกดปุ่ม
        prediction = model_ice.predict([[input_val]])
        
        
        
        # 3. ต้องย่อหน้าให้ st.markdown อยู่ข้างใน if เท่านั้น (สำคัญมาก!)
        st.markdown(f"""
            <div style="text-align: center; padding: 25px; border-radius: 15px; background-color: #e8f5e9; border: 2px solid #c8e6c9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 14px; color: #2e7d32; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">Predicted Sensor Value</p>
                <h1 style="margin: 10px 0; color: #1b5e20; font-size: 48px;">{prediction[0]:.4f}</h1>
                <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 15px 0;">
                <p style="margin: 0; font-size: 16px; color: #2e7d32; font-weight: bold;">
                </p>
                <div style="background-color: #ffffff; border-radius: 10px; height: 8px; margin-top: 10px;">
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.caption("<center style='margin-top:10px;'>Confidence score is based on Model performance during validation</center>", unsafe_allow_html=True)

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







