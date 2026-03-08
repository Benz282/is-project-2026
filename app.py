import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Project IS 2568", layout="wide")

# --- เมนูข้างๆ (Sidebar) ---
menu = st.sidebar.selectbox("เลือกหน้า", ["หน้าแรก", "ทดสอบโมเดล 1 (Ice)", "ทดสอบโมเดล 2 (MNIST)"])

# --- หน้าแรก ---
if menu == "หน้าแรก":
    st.title("🤖 โปรเจค AI - วิชา IS 2568")
    st.write("ยินดีต้อนรับสู่โปรเจคการพัฒนา Machine Learning และ Neural Network")
    st.info("กรุณาเลือกเมนูด้านซ้ายเพื่อเริ่มทดสอบโมเดล")

# --- หน้าทดสอบโมเดล 1 (Ensemble) ---
elif menu == "ทดสอบโมเดล 1 (Ice)":
    st.header("📊 ทดสอบโมเดล Ensemble (Ice Dataset)")
    
    # โหลดโมเดล
    model_ice = joblib.load('model_ice_ensemble.pkl')
    
    # สร้างช่องรับข้อมูลตัวเลข
    input_val = st.number_input("ใส่ค่า Timestamp เพื่อทำนายค่า Value", value=1540892278.0)
    
    if st.button("ทำนายผล"):
        # ทำนาย
        prediction = model_ice.predict([[input_val]])
        st.success(f"ผลการทำนายค่า Value คือ: {prediction[0]:.4f}")

# --- หน้าทดสอบโมเดล 2 (Neural Network) ---
elif menu == "ทดสอบโมเดล 2 (MNIST)":
    st.header("🧠 ทดสอบโมเดล Neural Network (จำแนกตัวเลข)")
    
    # โหลดโมเดล
    model_mnist = tf.keras.models.load_model('model_mnist_nn.h5')
    
    st.write("อัปโหลดรูปภาพตัวเลขเขียนมือ (พื้นหลังดำ เส้นสีขาว)")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # แสดงรูปที่อัปโหลด
        image = Image.open(uploaded_file).convert('L') # เปลี่ยนเป็นขาวดำ
        st.image(image, caption='รูปที่อัปโหลด', width=150)
        
        # เตรียมรูปให้ AI (ปรับขนาดเป็น 28x28)
        image = image.resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        if st.button("วิเคราะห์ตัวเลข"):
            # ทำนาย
            res = model_mnist.predict(img_array)
            final_res = np.argmax(res)
            st.success(f"AI คิดว่าตัวเลขนี้คือเลข: {final_res}")
