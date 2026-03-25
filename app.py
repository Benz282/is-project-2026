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
st.sidebar.title("Control Menu")
menu = st.sidebar.selectbox("เลือกหน้า", ["Homepage", "Model Project1 (Ice)", "Model Project 2 (MNIST)"])

# --- หน้าแรก ---
if menu == "Homepage":
    st.title("🤖 Project AI - Intelligent System 2568 ")
    st.write("""
             -**Welcome to the ML & Neural Network Development Project.**
             -**By** น.ส.โยษิตา คงธนประเสริฐ 6704062662194 Sec.5
             """)
    st.markdown("""
    ### Project Scope:
    - **Model 1:** Sensor Value Prediction on Ice Dataset using Ensemble Learning
    - **Model 2:** Handwritten Digit Classification (MNIST) using CNN
    """)
    st.info("💡 Select a menu from the sidebar to begin testing.")

# --- หน้าทดสอบโมเดล 1 (Ice) ---
elif menu == "Model Project1 (Ice)":
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
elif menu == "Model Project 2 (MNIST)":
    st.title("📊 Traditional ML vs. Neural Networks (MNIST)")
    st.caption("Project: Digit Recognizer (MNIST) | Model: Ensemble Voting Classifier & CNN")

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
        This model utilizes the **Convolutional Neural Network (CNN)** concept for high-accuracy image classification. 
        It extracts spatial features (edges, curves) to identify digits correctly regardless of handwriting style.
        """)

        st.subheader("3. Data References")
        st.success("🔗 **Source:** [Kaggle: Digit Recognizer (MNIST)](https://www.kaggle.com/c/digit-recognizer/data?select=sample_submission.csv)")

    # 2. ส่วนการทำนายผล (Prediction Section) - ย่อหน้าให้ตรงกับ elif ด้านบน
    st.divider()
    st.header("🔮 Model Prediction Testing")

    if model_mnist is None:
        st.error("❌ Model not found! Please ensure 'model_mnist_nn.h5' is loaded correctly.")
    else:
        # สร้างคอลัมน์เพื่อให้ UI ดูสมดุล
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("📤 Upload a handwritten digit...", type=["png", "jpg", "jpeg"], key="mnist_upload")
        
        if uploaded_file is not None:
            # โหลดรูปภาพ
            image = Image.open(uploaded_file).convert('L')
            
            # --- Preprocessing Step ---
            img_for_cv = np.array(image)
            # 1. ปรับปรุงรูปภาพ: หากพื้นหลังขาว ตัวเลขดำ ให้ Invert (สลับสี)
            if np.mean(img_for_cv) > 127: 
                from PIL import ImageOps
                image = ImageOps.invert(image)
                img_for_cv = np.array(image)
                 
            # 2. ตรวจสอบการหาตัวเลข (Contour Detection)
            _, thresh = cv2.threshold(img_for_cv, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_count = sum(1 for c in contours if cv2.contourArea(c) > 20)

            with col2:
                st.image(image, caption='Processed Image (28x28)', width=150)
            
            if digit_count > 1:
                st.error(f"⚠️ Found {digit_count} digits! Please upload only ONE digit at a time.")
            elif digit_count == 0:
                st.warning("❓ No digit detected. Try drawing more clearly.")
            else:
                # ปุ่มกดยืนยันการวิเคราะห์
                if st.button("🚀 Analyze Now", use_container_width=True):
                    with st.spinner('AI is thinking...'):
                        # 3. เตรียมข้อมูลเข้าโมเดล
                        img_resized = image.resize((28, 28))
                        img_array = np.array(img_resized)
                         
                        # Normalization
                        img_input = img_array.astype('float32') / 255.0
                        img_input = img_input.reshape(1, 28, 28, 1) # สำหรับ CNN
                         
                        # Predict
                        prediction = model_mnist.predict(img_input)
                        result = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        
                        # --- แสดงผลลัพธ์แบบตัวเลขขนาดใหญ่ ---
                        st.divider()
                        _, center_col, _ = st.columns([1, 2, 1])
                        
                        with center_col:
                            st.write("<p style='text-align: center; font-size: 20px;'>Predicted Digit</p>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style="background-color: #262730; border-radius: 10px; border: 2px solid #4CAF50; padding: 20px; text-align: center;"> 
                                    <h1 style="color: #4CAF50; font-size: 100px; margin: 0;">{result}</h1>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(int(confidence))
                            st.write(f"<p style='text-align: center;'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

                        if confidence > 80:
                            st.success("✅ Prediction Successful!")
                        else:
                            st.warning("⚠️ Low Confidence - The handwriting might be unclear.")
