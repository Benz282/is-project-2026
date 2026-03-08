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
    st.title("📊 การพัฒนาโมเดลทำนายค่าเซนเซอร์ (Ice Dataset)")
    
    # --- ส่วนอธิบายเนื้อหา (Documentation) ---
    with st.expander("📖 รายละเอียดแนวทางการพัฒนาโมเดล (คลิกเพื่ออ่าน)", expanded=True):
        st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
        st.write("""
        - **Data Cleaning:** ทำการตรวจสอบค่าว่าง (Missing Values) ใน Dataset หากพบจะใช้การแทนที่ด้วยค่าเฉลี่ย (Mean Imputation) เพื่อให้ข้อมูลมีความต่อเนื่อง
        - **Feature Selection:** คัดเลือกตัวแปร `Timestamp` เป็นตัวแปรต้น (X) เพื่อทำนายค่า `Value` (y) ซึ่งเป็นข้อมูลเชิงปริมาณ
        - **Data Splitting:** แบ่งข้อมูลออกเป็น Training Set 80% และ Test Set 20% เพื่อวัดประสิทธิภาพของโมเดล
        """)

        st.subheader("2. ทฤษฎีอัลกอริทึม (Ensemble Learning)")
        st.write("""
        โปรเจคนี้ใช้เทคนิค **Ensemble Voting Regressor** ซึ่งเป็นการรวมพลังของ 3 อัลกอริทึมหลัก ได้แก่:
        - **Linear Regression:** ใช้หาความสัมพันธ์เชิงเส้นพื้นฐาน
        - **Decision Tree:** ใช้การตัดสินใจแบบโครงสร้างต้นไม้เพื่อหาความสัมพันธ์ที่ซับซ้อน
        - **Random Forest:** ใช้การสร้างต้นไม้หลายต้นมาช่วยกันหาคำตอบ เพื่อลดการเกิด Overfitting
        - *หลักการ:* นำผลลัพธ์จากทั้ง 3 โมเดลมาหาค่าเฉลี่ย เพื่อให้ได้คำตอบที่แม่นยำและเสถียรที่สุด
        """)

        st.subheader("3. ขั้นตอนการพัฒนาโมเดล (Model Pipeline)")
        st.markdown("""
        1. นำเข้าข้อมูลจากไฟล์ `Ice.csv`
        2. จัดการข้อมูลเบื้องต้นและทำความสะอาดข้อมูลด้วย `Pandas`
        3. ฝึกสอนโมเดล (Training) ด้วย `Scikit-learn` โดยใช้เทคนิค Voting
        4. บันทึกโมเดลในรูปแบบไฟล์ `.pkl` เพื่อนำมาใช้บนเว็บไซต์
        """)

        st.subheader("4. แหล่งอ้างอิงข้อมูล (References)")
        st.info("📂 ข้อมูลชุดนี้อ้างอิงจาก: [Ice Dataset - Sensor Readings Samples] (แหล่งข้อมูลจำลองเพื่อการศึกษาภายในวิชา IS 2568)")

    st.divider()

    # --- ส่วนการทำนายผล (Prediction) ---
    st.subheader("🔮 ส่วนการทดสอบทำนายผล")
    if model_ice is None:
        st.error("❌ ไม่พบไฟล์ model_ice_ensemble.pkl กรุณาตรวจสอบการอัปโหลดไฟล์")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            input_val = st.number_input("ป้อนค่า Timestamp ที่ต้องการทดสอบ:", value=1540892278.0, format="%.1f")
        
        with col2:
            st.write("") # เว้นระยะ
            st.write("") # เว้นระยะ
            btn_predict = st.button("เริ่มการทำนายผล")

        if btn_predict:
            # ทำการทำนาย
            prediction = model_ice.predict([[input_val]])
            
            st.balloons()
            st.success(f"**ผลลัพธ์การทำนาย:** ค่า Value ที่ได้คือ **{prediction[0]:.4f}**")
            
            # ทำกราฟจำลองการทำนายเบื้องต้น
            st.info("💡 หมายเหตุ: ค่าที่ได้มาจากการคำนวณถ่วงน้ำหนักของโมเดล Ensemble ทั้ง 3 ตัว")
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

