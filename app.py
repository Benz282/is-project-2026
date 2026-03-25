import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps # เพิ่ม ImageOps สำหรับจัดการสีรูปภาพ
import cv2

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Project IS 2568", layout="wide")

# --- ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_all_models():
    try:
        # ตรวจสอบชื่อไฟล์โมเดลให้ตรงกับที่มีในโฟลเดอร์
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
    
    with st.expander("📖 Model Development Details", expanded=True):
        st.subheader("1. Data Preparation")
        st.write("- **Dataset:** Compass and sensor measurement data recorded during ice skating movements.")
        st.subheader("2. Algorithm Theory")
        st.write("Using **Voting Regressor** (Linear Regression, Decision Tree, Random Forest).")
        st.success("🔗 **Source:** [Kaggle: Ice Skating Data](https://www.kaggle.com/datasets/frankvanrest/ice-skating-compass-data)")

    st.divider()
    st.subheader("🔮 Model Prediction Testing")
    col_l, col_m, col_r = st.columns([1, 2, 1])

    with col_m:
        if model_ice is None:
            st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl")
        else:
            input_val = st.number_input("Enter Timestamp (Example: 1540892278):", value=1540892278.0, format="%.1f")
            if st.button("Run Prediction", use_container_width=True):
                prediction = model_ice.predict([[input_val]])
                st.snow()
                st.markdown(f"""
                    <div style="text-align: center; padding: 25px; border-radius: 15px; background-color: #e8f5e9; border: 2px solid #c8e6c9;">
                        <p style="color: #2e7d32; font-weight: bold;">Predicted Sensor Value</p>
                        <h1 style="color: #1b5e20; font-size: 48px;">{prediction[0]:.4f}</h1>
                    </div>
                """, unsafe_allow_html=True)

# --- หน้าทดสอบโมเดล 2 (MNIST) ---
elif menu == "ทดสอบโมเดล 2 (MNIST)":
    st.header("🧠 ทดสอบโมเดล Neural Network (MNIST)")
    st.title("📊 Traditional ML vs. Neural Networks (MNIST)")
    st.caption("Project: Digit Recognizer (MNIST) | Model: Ensemble Voting Classifier & CNN")
    
    with st.expander("📖 Model Development Details", expanded=True):
        st.subheader("1. Data Preparation")
        st.write("- **Normalization:** Scaled pixel values [0, 255] to [0, 1]")
        st.write("- **Independent Variable (X):** 784 Pixel Intensity Values")
        st.subheader("2. Algorithm Theory")
        st.write("Using **CNN** and **Voting Classifier** (Logistic Regression, RF, SVM).")
        st.success("🔗 **Source:** [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)")

    st.divider()
    st.subheader("🔮 Model Prediction Testing")

    if model_mnist is None:
        st.error("❌ Model not found! Please ensure 'model_mnist_nn.h5' is loaded correctly.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("📤 Upload a handwritten digit...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # 1. โหลดและจัดการรูปภาพ
            image = Image.open(uploaded_file).convert('L')
            img_for_cv = np.array(image)
            
            # ถ้าพื้นหลังขาว ให้ Invert เป็นพื้นดำตัวเลขขาวแบบ MNIST
            if np.mean(img_for_cv) > 127:
                image = ImageOps.invert(image)
                img_for_cv = np.array(image)

            # 2. ตรวจสอบว่ามีตัวเลขหรือไม่
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
                if st.button("🚀 Analyze Now", use_container_width=True):
                    with st.spinner('AI is thinking...'):
                        # เตรียมข้อมูลเข้าโมเดล
                        img_resized = image.resize((28, 28))
                        img_input = np.array(img_resized).astype('float32') / 255.0
                        img_input = img_input.reshape(1, 28, 28, 1) # สำหรับ CNN
                        
                        # ทำนาย
                        prediction = model_mnist.predict(img_input)
                        result = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        
                        # แสดงผลตัวเลขขนาดใหญ่
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
                            st.warning("⚠️ Low Confidence - Handwriting might be unclear.")

# --- Footer (แสดงทุกหน้า) ---
st.divider()
st.caption("Project: AI Analysis 2568 | Data Source: [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data)")
