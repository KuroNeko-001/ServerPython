import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
import datetime
import time
import signal
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# นำเข้าคลาส DataPreprocessor จากไฟล์ที่สร้างไว้
from data_preprocessor import DataPreprocessor

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_processing.log"),
        logging.StreamHandler()
    ]
)

# โหลดตัวแปรสภาพแวดล้อมจากไฟล์ .env (ถ้ามี)
load_dotenv()

# ===== การตั้งค่า =====
# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://edwardbat00147:O5DrPGWlRuaUKU4g@aqi-senors-rf.xk5y8sk.mongodb.net/")
SOURCE_DB = "CPE495final"
SOURCE_COLLECTION = "sensordatas"
TARGET_DB = "CPE495final"
TARGET_COLLECTION = "modelresults_engvers"

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "random_forest_model.joblib")
MODEL_COLUMNS_PATH = os.getenv("MODEL_COLUMNS_PATH", "model_columns.json")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "data_preprocessor.joblib")  # เพิ่มพาธของ Data Preprocessor

# สถานะการทำงาน
running = True

# ฟังก์ชันจัดการสัญญาณปิดโปรแกรม (Ctrl+C)
def signal_handler(sig, frame):
    global running
    print("\n⚠️ ได้รับสัญญาณหยุดการทำงาน กำลังปิดโปรแกรม...")
    running = False

# ลงทะเบียนตัวจัดการสัญญาณ
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== โหลดโมเดลและคอลัมน์ที่ใช้ในการทำนาย =====
def load_model():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_COLUMNS_PATH):
            logging.error(f"❌ ไม่พบไฟล์โมเดลหรือไฟล์คอลัมน์ โปรดรัน train_model_rf.py ก่อน")
            return None, None, None
            
        # โหลดโมเดล
        model = joblib.load(MODEL_PATH)
        
        # โหลดคอลัมน์
        with open(MODEL_COLUMNS_PATH, 'r') as f:
            model_columns = json.load(f)
            
        # โหลด Data Preprocessor (ถ้ามี)
        preprocessor = None
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logging.info("✅ โหลด Data Preprocessor สำเร็จ")
        else:
            logging.warning("⚠️ ไม่พบไฟล์ Data Preprocessor จะใช้เฉพาะโมเดลเท่านั้น")
            
        logging.info("✅ โหลดโมเดลสำเร็จ")
        return model, model_columns, preprocessor
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None, None

# ===== เชื่อมต่อกับ MongoDB =====
def connect_mongodb():
    try:
        # เชื่อมต่อกับ MongoDB Atlas
        logging.info(f"กำลังเชื่อมต่อกับ MongoDB Atlas...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการเชื่อมต่อ
        client.admin.command('ping')
        
        source_db = client[SOURCE_DB]
        source_collection = source_db[SOURCE_COLLECTION]
        
        target_db = client[TARGET_DB]
        target_collection = target_db[TARGET_COLLECTION]
        
        logging.info(f"✅ เชื่อมต่อกับ MongoDB Atlas สำเร็จ")
        return client, source_collection, target_collection
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อกับ MongoDB: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None

# ===== ฟังก์ชันทำนายคุณภาพอากาศจากข้อมูลเซ็นเซอร์ =====
def predict_air_quality(data, model, model_columns, preprocessor=None):
    try:
        # ปรับชื่อ field จาก ozone เป็น o3 ถ้ามี
        if 'ozone' in data and 'o3' not in data:
            data['o3'] = data['ozone']
        
        # แปลงหน่วย ppb เป็น ppm สำหรับ O3, NO2, SO2 ก่อนใช้กับโมเดล
        # โมเดลถูกเทรนด้วย ppm แต่ข้อมูลเซนเซอร์มาเป็น ppb
        if 'o3' in data and data['o3'] is not None:
            # บันทึกค่า ppb ไว้แสดงผล
            o3_ppb = data['o3']
            # แปลง ppb เป็น ppm (หาร 1000)
            data['o3'] = data['o3'] / 1000.0
            logging.info(f"แปลงค่า O3 จาก {o3_ppb} ppb เป็น {data['o3']} ppm สำหรับโมเดล")
        
        if 'no2' in data and data['no2'] is not None:
            # บันทึกค่า ppb ไว้แสดงผล
            no2_ppb = data['no2'] 
            # แปลง ppb เป็น ppm (หาร 1000)
            data['no2'] = data['no2'] / 1000.0
            logging.info(f"แปลงค่า NO2 จาก {no2_ppb} ppb เป็น {data['no2']} ppm สำหรับโมเดล")
            
        if 'so2' in data and data['so2'] is not None:
            # บันทึกค่า ppb ไว้แสดงผล
            so2_ppb = data['so2']
            # แปลง ppb เป็น ppm (หาร 1000)
            data['so2'] = data['so2'] / 1000.0
            logging.info(f"แปลงค่า SO2 จาก {so2_ppb} ppb เป็น {data['so2']} ppm สำหรับโมเดล")
        
        # ตรวจสอบว่ามีค่า PM2.5 หรือไม่ (เก็บไว้เพื่อบันทึกข้อมูลเท่านั้น)
        has_pm25 = 'pm2_5' in data and data['pm2_5'] is not None
        pm25_value = data.get('pm2_5', 0) if has_pm25 else 0
        
        # สร้าง DataFrame สำหรับการทำนายด้วยโมเดล ML
        input_df = pd.DataFrame([data])
        
        # ถ้ามี Data Preprocessor ให้ใช้เพื่อเพิ่มฟีเจอร์
        if preprocessor is not None:
            # ประมวลผลข้อมูลด้วย Data Preprocessor
            logging.info("ใช้ Data Preprocessor เพื่อเพิ่มฟีเจอร์")
            input_df = preprocessor.process_data(input_df)
            logging.info(f"เพิ่มฟีเจอร์สำเร็จ: {list(input_df.columns)}")
        
        # เก็บค่าที่ประมาณอย่างชาญฉลาด
        estimated_values = {}
        
        # เพิ่มค่า CO2 เข้าไปเพื่อให้โมเดลทำงานได้ (แต่ไม่ได้ใช้จริง)
        if 'co2' in model_columns and 'co2' not in input_df.columns:
            # ใส่ค่า default เป็น 0 ซึ่งจะมีผลน้อยที่สุดกับโมเดล
            input_df['co2'] = 0
            logging.info("ℹ️ จำเป็นต้องใส่ค่า CO2=0 สำหรับโมเดล (แต่ไม่ได้ใช้ในระบบจริง)")
        
        # ประมาณค่าที่ขาดหายไปอย่างชาญฉลาด
        missing_columns = []
        for col in model_columns:
            # ข้ามการประมาณค่า CO2
            if col == 'co2':
                continue
                
            if col not in input_df.columns:
                missing_columns.append(col)
                
                # ประมาณค่าที่ขาดหายไปอย่างชาญฉลาด
                if col == 'pm10' and 'pm2_5' in input_df.columns:
                    # PM10 มักมีค่าประมาณ 1.85-2.2 เท่าของ PM2.5
                    input_df[col] = input_df['pm2_5'] * 1.85
                    estimated_values[col] = float(input_df[col].values[0])  # เก็บค่าที่ประมาณการ
                elif col == 'co':
                    # ค่า CO เฉลี่ยในห้องเรียน
                    input_df[col] = 1.0  # ค่าปกติประมาณ 0.5-2 ppm
                    estimated_values[col] = float(input_df[col].values[0])  # เก็บค่าที่ประมาณการ
                elif col == 'o3':
                    # ตรวจสอบว่ามี ozone ในข้อมูลหรือไม่
                    if 'ozone' in input_df.columns:
                        input_df[col] = input_df['ozone'] / 1000.0  # แปลง ppb เป็น ppm
                        estimated_values[col] = float(input_df[col].values[0] * 1000.0)  # เก็บค่าที่ประมาณการเป็น ppb
                    elif 'temperature' in input_df.columns and 'humidity' in input_df.columns:
                        # O3 มีความสัมพันธ์กับอุณหภูมิและความชื้น (โดยประมาณ)
                        temp_factor = (input_df['temperature'] - 20) / 15  # ผลกระทบจากอุณหภูมิที่ปรับให้เป็นมาตรฐาน
                        humid_factor = (70 - input_df['humidity']) / 50  # ความชื้นมีความสัมพันธ์แบบผกผัน
                        
                        # คำนวณเป็น ppm ก่อน (สำหรับโมเดล)
                        o3_ppm = 0.05 + (temp_factor * 0.05) + (humid_factor * 0.03)
                        o3_ppm = np.clip(o3_ppm, 0.01, 0.2)  # จำกัดให้อยู่ในช่วงที่เหมาะสม
                        
                        # แปลงเป็น ppb สำหรับการเก็บค่าประมาณการ (1 ppm = 1000 ppb)
                        o3_ppb = o3_ppm * 1000.0
                        
                        # ใช้ค่า ppm กับโมเดล
                        input_df[col] = o3_ppm
                        estimated_values[col] = float(o3_ppb)  # เก็บค่าที่ประมาณการเป็น ppb
                        # แก้ไขจาก Series เป็นค่าเดียว
                        o3_ppb_val = float(o3_ppb)
                        o3_ppm_val = float(o3_ppm)
                        logging.info(f"ประมาณค่า O3: {o3_ppb_val:.2f} ppb ({o3_ppm_val:.5f} ppm)")
                elif col == 'no2':
                    # ค่า NO2 เฉลี่ยในห้องเรียน (0.02 ppm = 20 ppb)
                    input_df[col] = 0.02  # ค่าปกติประมาณ 0.01-0.05 ppm (สำหรับโมเดล)
                    estimated_values[col] = 20.0  # เก็บค่าที่ประมาณการเป็น ppb
                    logging.info(f"ประมาณค่า NO2: 20.00 ppb (0.02000 ppm)")
                elif col == 'so2':
                    # SO2 มักมีค่าต่ำในอาคารที่ไม่มีแหล่งกำเนิดโดยตรง (0.02 ppm = 20 ppb)
                    input_df[col] = 0.02  # ppm (สำหรับโมเดล)
                    estimated_values[col] = 20.0  # เก็บค่าที่ประมาณการเป็น ppb
                    logging.info(f"ประมาณค่า SO2: 20.00 ppb (0.02000 ppm)")
                elif col == 'pm2_5':
                    # ตรวจสอบชื่อทางเลือกสำหรับ PM2.5
                    pm25_alternatives = ['pm25', 'PM2.5', 'PM25', 'PM_2_5', 'pm_2_5']
                    found_alt = False
                    
                    # ตรวจสอบชื่อทางเลือก
                    for alt in pm25_alternatives:
                        if alt in input_df.columns:
                            input_df[col] = input_df[alt]
                            estimated_values[col] = float(input_df[col].values[0])
                            found_alt = True
                            break
                    
                    # ถ้าไม่พบทางเลือก ให้ใช้การประมาณค่า
                    if not found_alt:
                        if 'pm10' in input_df.columns:
                            # วิธีที่ 1: ประมาณ PM2.5 จาก PM10
                            # สูตร: PM2.5 = PM10 หารด้วย 1.85
                            # เช่น ถ้า PM10 = 92.5, PM2.5 จะประมาณ = 50
                            input_df[col] = input_df['pm10'] / 1.85
                            estimated_values[col] = float(input_df[col].values[0])
                        elif 'temperature' in input_df.columns and 'humidity' in input_df.columns:
                            # วิธีที่ 2: ประมาณ PM2.5 จากอุณหภูมิและความชื้น
                            # แนวคิด: 
                            # - อุณหภูมิไม่เหมาะสม (ร้อนหรือเย็นเกิน) มักทำให้ฝุ่นลอยตัวได้ไม่ดี
                            # - ความชื้นผิดปกติ ส่งผลต่อการลอยตัวของฝุ่น
                            
                            # คำนวณปัจจัยแต่ละตัว
                            temp_factor = np.abs(input_df['temperature'] - 25) / 10  # ห่างจาก 25°C
                            humid_factor = np.clip(np.abs(input_df['humidity'] - 50) / 20, 0, 2)  # ห่างจากความชื้น 50%
                            
                            # สูตรประมาณค่า PM2.5 จากปัจจัยอุณหภูมิและความชื้น
                            base_pm25 = 15.0  # ค่าฐานในห้องทั่วไป
                            pm25_estimate = base_pm25 + (humid_factor * 8.0) + (temp_factor * 5.0)
                            
                            # ปรับแต่งตามสถานที่ (เป็นห้องเรียน)
                            classroom_factor = 1.2  # ห้องเรียนมักมีค่า PM2.5 สูงกว่าปกติเล็กน้อย
                            input_df[col] = pm25_estimate * classroom_factor
                            
                            # จำกัดให้อยู่ในช่วงที่สมเหตุสมผล
                            input_df[col] = np.clip(input_df[col], 5.0, 150.0)
                            estimated_values[col] = float(input_df[col].values[0])
                        else:
                            # วิธีที่ 3: ถ้าไม่มีข้อมูลเพียงพอ ใช้ค่าเฉลี่ยตามฤดูกาล
                            # ตรวจสอบเดือนปัจจุบัน (ถ้ามี timestamp)
                            current_month = datetime.datetime.now().month
                            if 'timestamp' in data and isinstance(data['timestamp'], datetime.datetime):
                                current_month = data['timestamp'].month
                                
                            # ปรับค่าตามฤดูกาล (สำหรับประเทศไทย)
                            if 11 <= current_month <= 2:  # ฤดูหนาว (พ.ย.-ก.พ.) - มักมีมลพิษสูง
                                input_df[col] = 35.0
                            elif 3 <= current_month <= 5:  # ฤดูร้อน (มี.ค.-พ.ค.) - ฝุ่นหมอกควัน
                                input_df[col] = 40.0
                            elif 6 <= current_month <= 10:  # ฤดูฝน (มิ.ย.-ต.ค.) - มลพิษต่ำกว่า
                                input_df[col] = 18.0
                            else:
                                input_df[col] = 25.0  # ค่าเฉลี่ยทั่วไป
                            estimated_values[col] = float(input_df[col].values[0])
                elif col == 'pm10':
                    # ตรวจสอบชื่อทางเลือกสำหรับ PM10
                    pm10_alternatives = ['PM10', 'PM_10', 'pm_10']
                    found_alt = False
                    
                    # ตรวจสอบชื่อทางเลือก
                    for alt in pm10_alternatives:
                        if alt in input_df.columns:
                            input_df[col] = input_df[alt]
                            estimated_values[col] = float(input_df[col].values[0])
                            found_alt = True
                            break
                    
                    # ถ้าไม่พบทางเลือก ให้ใช้การประมาณค่า
                    if not found_alt:
                        if 'pm2_5' in input_df.columns:
                            # ประมาณค่า PM10 จาก PM2.5
                            # PM10 มักมีค่าประมาณ 1.7-2.1 เท่าของ PM2.5 ขึ้นอยู่กับสภาพแวดล้อม
                            # ใช้ค่าเฉลี่ยสำหรับในร่ม (ห้องเรียน)
                            input_df[col] = input_df['pm2_5'] * 1.85  # ค่าเฉลี่ยทั่วไป
                            estimated_values[col] = float(input_df[col].values[0])
                        elif 'temperature' in input_df.columns and 'humidity' in input_df.columns:
                            # ประมาณค่า PM10 จากอุณหภูมิและความชื้น โดยคำนวณ PM2.5 ก่อน แล้วจึงคำนวณ PM10
                            temp_factor = np.abs(input_df['temperature'] - 25) / 10
                            humid_factor = np.clip(np.abs(input_df['humidity'] - 50) / 20, 0, 2)
                            
                            base_pm25 = 15.0  # ค่าฐานในห้องทั่วไป
                            pm25_estimate = base_pm25 + (humid_factor * 8.0) + (temp_factor * 5.0)
                            
                            # ประมาณค่า PM10 จาก PM2.5 ที่คำนวณได้
                            # ใช้ค่าเฉลี่ยสำหรับในร่ม (ห้องเรียน)
                            pm10_factor = 1.85  # ใช้ค่าเฉลี่ยทั่วไป
                                
                            input_df[col] = pm25_estimate * pm10_factor
                            
                            # จำกัดให้อยู่ในช่วงที่สมเหตุสมผล
                            input_df[col] = np.clip(input_df[col], 10.0, 250.0)
                            estimated_values[col] = float(input_df[col].values[0])
                        else:
                            # ถ้าไม่มีข้อมูลเพียงพอ ใช้ค่าเฉลี่ยที่ดีขึ้น ซึ่งอาจแตกต่างกันตามฤดูกาล
                            current_month = datetime.datetime.now().month
                            if 'timestamp' in data and isinstance(data['timestamp'], datetime.datetime):
                                current_month = data['timestamp'].month
                                
                            # ปรับค่าตามฤดูกาล (สำหรับประเทศไทย)
                            if 11 <= current_month <= 2:  # ฤดูหนาว - มักมีมลพิษสูง
                                input_df[col] = 65.0
                            elif 3 <= current_month <= 5:  # ฤดูร้อน - ฝุ่นหมอกควัน
                                input_df[col] = 75.0
                            elif 6 <= current_month <= 10:  # ฤดูฝน - มลพิษต่ำกว่า
                                input_df[col] = 35.0
                            else:
                                input_df[col] = 45.0  # ค่าเฉลี่ยทั่วไป
                            estimated_values[col] = float(input_df[col].values[0])
                else:
                    # สำหรับคอลัมน์อื่นๆ ใช้ค่าเริ่มต้นที่เหมาะสม
                    if col == 'temperature':
                        input_df[col] = 25.0  # อุณหภูมิห้องเฉลี่ย
                    elif col == 'humidity':
                        input_df[col] = 60.0  # ความชื้นห้องเฉลี่ย
                    else:
                        input_df[col] = 0.0  # ค่าเริ่มต้นสำหรับคอลัมน์อื่นๆ
                        estimated_values[col] = float(input_df[col].values[0])  # เก็บค่าที่ประมาณการ
                
        # แสดงคำเตือนถ้าข้อมูลขาดหายไป
        if missing_columns:
            missing_msg = f"⚠️ ข้อมูลบางส่วนขาดหายและได้รับการประมาณโดยอัตโนมัติ: {', '.join(missing_columns)}"
            logging.warning(missing_msg)
        
        # จัดเรียงคอลัมน์ให้ตรงกับโมเดล
        input_df = input_df[model_columns]
        
        # ========================= ขั้นตอนการทำนาย =========================
        # ทำนายด้วยโมเดล ML
        model_prediction = None
        prediction_method = "model"  # เริ่มต้น ใช้โมเดลเป็นหลัก
        probabilities = None
        
        try:
            model_prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
        except Exception as e:
            logging.error(f"❌ เกิดข้อผิดพลาดในการทำนายด้วยโมเดล: {str(e)}")
            model_prediction = None
        
        # ตัดสินใจว่าจะใช้วิธีการทำนายแบบใด
        if model_prediction is not None:
            # ใช้การทำนายจากโมเดล ML เป็นหลัก
            final_prediction = model_prediction
            prediction_method = "model"
            
            # บันทึกค่า PM2.5 เพื่อเป็นข้อมูลเพิ่มเติม
            if has_pm25:
                logging.info(f"ℹ️ ค่า PM2.5 = {pm25_value} μg/m³ | การทำนายจากโมเดล = ระดับ {model_prediction}")
                
            # ตรวจสอบอุณหภูมิและปรับระดับคุณภาพอากาศเพิ่มเติม
            if 'temperature' in data:
                temp = data['temperature']
                original_prediction = final_prediction
                
                # ตรวจสอบอุณหภูมิตามเงื่อนไขและปรับระดับคุณภาพอากาศ
                if temp >= 40:  # อันตราย (ร้อนจัด)
                    final_prediction = 4  # ต้องเป็นระดับแย่ที่สุด
                    prediction_method = "temp_override"
                    logging.info(f"⚠️ อุณหภูมิสูงมาก ({temp}°C) - ปรับระดับเป็น 4 (แย่ที่สุด)")
                elif temp >= 38:  # ไม่ดีต่อสุขภาพ (ร้อนมาก)
                    final_prediction = max(final_prediction, 3)
                    if final_prediction > original_prediction:
                        prediction_method = "temp_adjusted"
                        logging.info(f"⚠️ อุณหภูมิร้อนมาก ({temp}°C) - ปรับระดับเป็น {final_prediction}")
                elif temp >= 35:  # มีผลต่อคนอ่อนไหว (ร้อน)
                    final_prediction = max(final_prediction, 2)
                    if final_prediction > original_prediction:
                        prediction_method = "temp_adjusted"
                        logging.info(f"ℹ️ อุณหภูมิค่อนข้างร้อน ({temp}°C) - ปรับระดับเป็น {final_prediction}")
                elif temp <= 10:  # หนาวมาก
                    final_prediction = max(final_prediction, 3)
                    if final_prediction > original_prediction:
                        prediction_method = "temp_adjusted"
                        logging.info(f"⚠️ อุณหภูมิหนาวมาก ({temp}°C) - ปรับระดับเป็น {final_prediction}")
                elif temp <= 15:  # หนาวเกินไป
                    final_prediction = max(final_prediction, 2)
                    if final_prediction > original_prediction:
                        prediction_method = "temp_adjusted"
                        logging.info(f"ℹ️ อุณหภูมิหนาว ({temp}°C) - ปรับระดับเป็น {final_prediction}")
                        
                # ถ้ามีฟีเจอร์ thermal_discomfort ให้ใช้ด้วย
                if 'thermal_discomfort' in input_df.columns:
                    discomfort = input_df['thermal_discomfort'].values[0]
                    if discomfort >= 4:  # เครียดจากความร้อนรุนแรงหรืออันตรายถึงชีวิต
                        final_prediction = max(final_prediction, 4)
                        if final_prediction > original_prediction:
                            prediction_method = "temp_adjusted"
                            logging.info(f"⚠️ ความเครียดจากความร้อนอันตราย (ระดับ {discomfort}) - ปรับเป็นระดับ {final_prediction}")
                    elif discomfort >= 3:  # เครียดจากความร้อน
                        final_prediction = max(final_prediction, 3)
                        if final_prediction > original_prediction:
                            prediction_method = "temp_adjusted"
                            logging.info(f"⚠️ ความเครียดจากความร้อนสูง (ระดับ {discomfort}) - ปรับเป็นระดับ {final_prediction}")
        else:
            # ถ้าไม่สามารถทำนายด้วยโมเดลได้ ใช้ค่า fallback
            final_prediction = 2  # ค่าเฉลี่ย (ระดับปานกลาง)
            prediction_method = "fallback"
            logging.warning(f"⚠️ ไม่สามารถทำนายด้วยโมเดล ML ได้ ใช้ค่าระดับปานกลาง (ระดับ 2)")
            if has_pm25:
                logging.info(f"ℹ️ ค่า PM2.5 = {pm25_value} μg/m³")
        
        # ระดับคุณภาพอากาศตามมาตรฐานของไทย 
        aqi_labels = {
            0: "Excellent Air Quality (0-25 μg/m³)",
            1: "Good Air Quality (26-37 μg/m³)",
            2: "Moderate Air Quality (38-50 μg/m³)",
            3: "Unhealthy for Sensitive Groups (51-90 μg/m³)",
            4: "Unhealthy Air Quality (91+ μg/m³)"
        }
        
        # เพิ่มข้อมูลความเสี่ยงต่อสุขภาพสำหรับแต่ละระดับ (กรณีมลพิษเป็นสาเหตุหลัก) 
        health_risks_pollution = {
            0: "Very Low Risk: Excellent classroom air quality promotes learning and student development. No impact on concentration or learning efficiency.",
            1: "Low Risk: Good air quality with minimal effects on students with allergies or respiratory issues. Still suitable for normal teaching activities.",
            2: "Moderate Risk: May cause eye, nose, and throat irritation for some students, which might slightly reduce concentration during classes.",
            3: "High Risk: Students may experience reduced attention span, fatigue, or headaches due to air pollution. Directly impacts learning efficiency.",
            4: "Very High Risk: Most students will be severely affected with fatigue, difficulty breathing, inability to concentrate, and may require suspension of teaching activities due to poor air quality."
        }
        
        # เพิ่มข้อมูลความเสี่ยงต่อสุขภาพสำหรับกรณีอุณหภูมิสูงเกินไป
        health_risks_high_temp = {
            0: "Very Low Risk: Comfortable temperature promotes good learning environment.",
            1: "Low Risk: Slightly warm but still comfortable for most students.",
            2: "Moderate Risk: Elevated temperature may cause discomfort and mild fatigue for some students, possibly affecting concentration.",
            3: "High Risk: High temperature causing significant discomfort, excessive sweating, and fatigue. Students may experience headaches, dizziness, and severely reduced ability to concentrate.",
            4: "Very High Risk: Dangerous heat levels posing risk of heat exhaustion or heat stroke. Not suitable for educational activities. Immediate cooling measures required."
        }
        
        # เพิ่มข้อมูลความเสี่ยงต่อสุขภาพสำหรับกรณีอุณหภูมิต่ำเกินไป
        health_risks_low_temp = {
            0: "Very Low Risk: Comfortable temperature promotes good learning environment.",
            1: "Low Risk: Slightly cool but still comfortable with proper clothing.",
            2: "Moderate Risk: Cool environment may cause discomfort for some students, affecting ability to focus in class.",
            3: "High Risk: Cold temperatures significantly affecting comfort and concentration. Students may experience cold extremities and difficulty writing.",
            4: "Very High Risk: Extremely cold conditions inadequate for learning activities. Risk of hypothermia and cold-related health issues."
        }
        
        # เพิ่มคำแนะนำสำหรับแต่ละระดับ (กรณีมลพิษเป็นสาเหตุหลัก) 
        health_recommendations_pollution = {
            0: "Continue classroom activities as normal. This environment is ideal for all forms of teaching and learning.",
            1: "Monitor students with allergies or respiratory issues. Normal classroom activities can continue.",
            2: "Consider using air purifiers in the classroom, reduce dust-generating activities, and avoid opening windows when outdoor air quality is poor.",
            3: "Use air purifiers continuously, consider reducing physical activities, monitor students with health issues, and consider shorter lesson times with more breaks.",
            4: "Consider suspending classes in this room or relocating to an area with better air quality. Students should wear N95 masks in the classroom, use multiple air purifiers, and closely monitor students with health issues."
        }
        
        # เพิ่มคำแนะนำสำหรับกรณีอุณหภูมิสูงเกินไป
        health_recommendations_high_temp = {
            0: "Continue classroom activities as normal. Maintain current temperature settings.",
            1: "Ensure adequate ventilation. Consider using fans if needed for air circulation.",
            2: "Use fans and air conditioning if available. Provide access to drinking water. Reduce physical activities during warmer parts of the day.",
            3: "Ensure air conditioning is working properly. Provide cool drinking water. Avoid physical activities. Consider shortened class periods with more breaks.",
            4: "Relocate classes to cooler areas if possible. If not, cancel classes until temperature decreases. Provide cooling stations and monitor students for signs of heat exhaustion."
        }
        
        # เพิ่มคำแนะนำสำหรับกรณีอุณหภูมิต่ำเกินไป
        health_recommendations_low_temp = {
            0: "Continue classroom activities as normal. Maintain current temperature settings.",
            1: "Ensure room is properly heated. Allow students to wear additional layers if needed.",
            2: "Increase heating if possible. Encourage warm clothing layers. Provide warm beverages if appropriate.",
            3: "Ensure adequate heating and insulation. Consider moving to warmer rooms. Allow warm clothing and blankets. Provide warm beverages.",
            4: "Relocate classes to properly heated areas. If not possible, cancel classes until adequate heating is available. Ensure students have proper cold-weather protection."
        }
        
        # เลือกชุดข้อความที่เหมาะสมตามสาเหตุ
        selected_health_risk = ""
        selected_health_recommendation = ""
        
        if prediction_method == "temp_override" or prediction_method == "temp_adjusted":
            # ตรวจสอบว่าเป็นปัญหาอุณหภูมิสูงหรือต่ำ
            if 'temperature' in data:
                temp = data['temperature']
                if temp >= 35:  # อุณหภูมิสูง
                    selected_health_risk = health_risks_high_temp[final_prediction]
                    selected_health_recommendation = health_recommendations_high_temp[final_prediction]
                    logging.info("ใช้ข้อความสำหรับอุณหภูมิสูง")
                elif temp <= 15:  # อุณหภูมิต่ำ
                    selected_health_risk = health_risks_low_temp[final_prediction]
                    selected_health_recommendation = health_recommendations_low_temp[final_prediction]
                    logging.info("ใช้ข้อความสำหรับอุณหภูมิต่ำ")
                else:
                    # ถ้าอุณหภูมิยังอยู่ในช่วงปกติ แต่มีการปรับค่า ให้ใช้ข้อความเกี่ยวกับมลพิษ
                    selected_health_risk = health_risks_pollution[final_prediction]
                    selected_health_recommendation = health_recommendations_pollution[final_prediction]
                    logging.info("ใช้ข้อความสำหรับมลพิษ เนื่องจากอุณหภูมิอยู่ในช่วงปกติ")
            else:
                # ถ้าไม่มีข้อมูลอุณหภูมิ ให้ใช้ข้อความเกี่ยวกับมลพิษ
                selected_health_risk = health_risks_pollution[final_prediction]
                selected_health_recommendation = health_recommendations_pollution[final_prediction]
                logging.info("ใช้ข้อความสำหรับมลพิษ เนื่องจากไม่มีข้อมูลอุณหภูมิ")
        else:
            # กรณีทำนายจากโมเดลปกติ ให้ใช้ข้อความเกี่ยวกับมลพิษ
            selected_health_risk = health_risks_pollution[final_prediction]
            selected_health_recommendation = health_recommendations_pollution[final_prediction]
            logging.info("ใช้ข้อความสำหรับมลพิษ เนื่องจากเป็นการทำนายจากโมเดลปกติ")
            
        # ปรับความน่าจะเป็นให้สอดคล้องกับวิธีการทำนาย
        if probabilities is None or len(probabilities) < 5:
            # สร้างความน่าจะเป็นใหม่
            adjusted_probs = [0.0] * 5
            
            if prediction_method == "fallback":
                # กรณีข้อมูลไม่เพียงพอ ความเชื่อมั่นต่ำ
                adjusted_probs[2] = 0.5  # ปานกลาง
                adjusted_probs[1] = 0.25  # ดี
                adjusted_probs[3] = 0.25  # เริ่มมีผลกระทบ
            else:
                # กรณีใช้โมเดลอย่างเดียว แต่ไม่มีค่าความน่าจะเป็น
                adjusted_probs[final_prediction] = 0.7  # ให้น้ำหนักกับคลาสที่ทำนาย
                
                # กระจายความน่าจะเป็นไปยังคลาสใกล้เคียง
                if final_prediction > 0:
                    adjusted_probs[final_prediction-1] = 0.15
                if final_prediction < 4:
                    adjusted_probs[final_prediction+1] = 0.15
                    
                # กระจายความน่าจะเป็นที่เหลือ (ถ้ามี)
                remaining = 1.0 - sum(adjusted_probs)
                if remaining > 0:
                    for i in range(5):
                        if adjusted_probs[i] == 0.0:
                            adjusted_probs[i] = remaining / (5 - sum(1 for p in adjusted_probs if p > 0))
        else:
            # ใช้ความน่าจะเป็นจากโมเดลโดยตรง
            adjusted_probs = probabilities.tolist()
        
        # 7. สร้าง dictionary ผลลัพธ์
        result = {
            "aqi_class": int(final_prediction),
            "aqi_label": aqi_labels[final_prediction],
            "health_risk": selected_health_risk,
            "health_recommendation": selected_health_recommendation,
            "method": prediction_method, 
            "probabilities": {f"class_{i}": float(prob) for i, prob in enumerate(adjusted_probs)},
            "estimated_values": estimated_values  # เพิ่มค่าที่ประมาณการ
        }

        # 8. เพิ่มข้อมูลเซ็นเซอร์ที่สำคัญ
        important_sensors = {}
        
        if 'pm2_5' in data:
            important_sensors['pm2_5'] = {"value": data['pm2_5'], "unit": "μg/m³"}
        if 'pm10' in data:
            important_sensors['pm10'] = {"value": data['pm10'], "unit": "μg/m³"}
        if 'co' in data:
            important_sensors['co'] = {"value": data['co'], "unit": "ppm"}
        if 'temperature' in data:
            important_sensors['temperature'] = {"value": data['temperature'], "unit": "°C"}
        if 'humidity' in data:
            important_sensors['humidity'] = {"value": data['humidity'], "unit": "%"}
            
        # เพิ่มข้อมูลเซ็นเซอร์อื่นๆ ที่สำคัญ
        if 'so2' in data:
            important_sensors['so2'] = {"value": data['so2'], "unit": "ppb"}
        if 'o3' in data:
            important_sensors['o3'] = {"value": data['o3'], "unit": "ppb"}
        if 'no2' in data:
            important_sensors['no2'] = {"value": data['no2'], "unit": "ppb"}
            
        # เพิ่มข้อมูลเซ็นเซอร์ลงในผลลัพธ์
        result["key_sensors"] = important_sensors
        
        # ===== ปรับข้อมูลกลับไปเป็นค่าดั้งเดิมหลังจากใช้โมเดล =====
        # บันทึกค่า ppb กลับเข้าไปในข้อมูลเดิม (ถ้ามี)
        if 'o3' in data and data['o3'] is not None and 'o3_ppb' in locals():
            data['o3'] = o3_ppb  # ใช้ค่า ppb ที่เก็บไว้ตอนแรก
            logging.info(f"คืนค่า O3 เป็น {o3_ppb} ppb สำหรับการแสดงผล")
        
        if 'no2' in data and data['no2'] is not None and 'no2_ppb' in locals():
            data['no2'] = no2_ppb  # ใช้ค่า ppb ที่เก็บไว้ตอนแรก
            logging.info(f"คืนค่า NO2 เป็น {no2_ppb} ppb สำหรับการแสดงผล")
        
        if 'so2' in data and data['so2'] is not None and 'so2_ppb' in locals():
            data['so2'] = so2_ppb  # ใช้ค่า ppb ที่เก็บไว้ตอนแรก
            logging.info(f"คืนค่า SO2 เป็น {so2_ppb} ppb สำหรับการแสดงผล")
        
        return result
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def process_one_document(item, model, model_columns, target_collection, preprocessor=None):
    """ประมวลผลข้อมูลเซ็นเซอร์แต่ละรายการ"""
    try:
        # สร้างข้อมูลเซ็นเซอร์
        sensor_data = {}
        
        # แปลงข้อมูลตามที่มีในคอลเลกชัน sensordatas
        for key in item:
            # ไม่รวม CO2 
            if key not in ['_id', 'timestamp', '__v', 'co2']:
                sensor_data[key] = item[key]
        
        # เพิ่ม timestamp
        if 'timestamp' in item:
            sensor_data['timestamp'] = item['timestamp']
        
        # ตรวจสอบว่ามี pm2_5 หรือไม่ ถ้าไม่มีให้ประมาณค่าโดยตรงก่อนเรียก predict_air_quality
        has_pm25 = 'pm2_5' in sensor_data or any(key in sensor_data for key in ['pm25', 'PM2.5', 'PM25', 'PM_2_5', 'pm_2_5'])
        
        if not has_pm25 and 'pm10' in sensor_data:
            # ถ้าไม่มี pm2_5 แต่มี pm10 ให้ประมาณค่า pm2_5 จาก pm10
            sensor_data['pm2_5'] = sensor_data['pm10'] / 1.85
            logging.info(f"ประมาณค่า pm2_5 = {sensor_data['pm2_5']} μg/m³ จาก pm10 = {sensor_data['pm10']} μg/m³")
        
        # ทำนายคุณภาพอากาศ (ส่ง preprocessor ไปด้วย)
        prediction_result = predict_air_quality(sensor_data, model, model_columns, preprocessor)
        
        if prediction_result:
            # สร้างข้อมูลดิบทั้งหมดที่รับมา (ยกเว้น timestamp และ __v ซึ่งจะเพิ่มทีหลัง)
            raw_data = {}
            timestamp_value = None
            v_value = None
            
            # เก็บค่า timestamp และ __v ไว้ก่อน
            if 'timestamp' in item:
                timestamp_value = item['timestamp']
            if '__v' in item:
                v_value = item['__v']
            
            # เพิ่มข้อมูลอื่นๆ ที่ไม่ใช่ timestamp และ __v และไม่ใช่ co2
            for key in item:
                if key not in ['_id', 'timestamp', '__v', 'co2']:
                    raw_data[key] = item[key]
            
            # เพิ่มค่าที่ประมาณอย่างชาญฉลาดลงใน raw_data
            if 'estimated_values' in prediction_result:
                # รวมค่าที่ประมาณการเข้ากับ raw_data
                for key, value in prediction_result['estimated_values'].items():
                    if key not in raw_data:  # เพิ่มเฉพาะกรณีที่ยังไม่มีข้อมูลนี้
                        raw_data[key] = value
                        
                # ตรวจสอบว่ามี pm2_5 หรือไม่ หากไม่มีให้เพิ่มอีกครั้ง
                if 'pm2_5' not in raw_data and 'pm10' in raw_data:
                    raw_data['pm2_5'] = raw_data['pm10'] / 1.85
                    logging.info(f"เพิ่มค่า pm2_5 = {raw_data['pm2_5']} μg/m³ ใน raw_data")
                
                # ลบ estimated_values ออกจาก prediction_result เพื่อไม่ให้เกิดความซ้ำซ้อน
                del prediction_result['estimated_values']
            
            # เพิ่ม timestamp และ __v ต่อท้าย (เพื่อให้อยู่ด้านล่างสุด)
            if timestamp_value is not None:
                raw_data['timestamp'] = timestamp_value
            if v_value is not None:
                raw_data['__v'] = v_value
            
            # สร้างเอกสารที่จะบันทึก
            document = {
                "timestamp": item.get('timestamp', datetime.datetime.now()),
                "date": item.get('timestamp', datetime.datetime.now()).strftime("%Y-%m-%d") if isinstance(item.get('timestamp'), datetime.datetime) else datetime.datetime.now().strftime("%Y-%m-%d"),
                "time": item.get('timestamp', datetime.datetime.now()).strftime("%H:%M:%S") if isinstance(item.get('timestamp'), datetime.datetime) else datetime.datetime.now().strftime("%H:%M:%S"),
                "sensor_data": sensor_data,
                "prediction": prediction_result,
                "source_id": item.get('_id'),
                "raw_data": raw_data  # เพิ่มข้อมูลดิบทั้งหมดรวมค่าที่ประมาณการ
            }
            
            # บันทึกลงใน MongoDB
            result = target_collection.insert_one(document)
            
            # แสดงผลลัพธ์
            logging.info(f"✅ ประมวลผลและบันทึกข้อมูล ID: {result.inserted_id}")
            logging.info(f"   → ระดับคุณภาพอากาศ: {prediction_result['aqi_class']} - {prediction_result['aqi_label']}")
            return True
        else:
            logging.warning(f"⚠️ ไม่สามารถทำนายข้อมูล ID: {item.get('_id')}")
            return False
    
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def process_sensors_data():
    """ดึงข้อมูลจาก sensordatas มาประมวลผลและบันทึกใน Model_Result (รูปแบบเดิม)"""
    
    # โหลดโมเดล และ Preprocessor
    model, model_columns, preprocessor = load_model()
    if model is None or model_columns is None:
        logging.error("❌ ไม่สามารถโหลดโมเดลได้ ยกเลิกการประมวลผล")
        return
    
    # เชื่อมต่อกับ MongoDB
    client, source_collection, target_collection = connect_mongodb()
    if None in (client, source_collection, target_collection):
        logging.error("❌ ไม่สามารถเชื่อมต่อกับ MongoDB ได้ ยกเลิกการประมวลผล")
        return
    
    try:
        # ดึงข้อมูลเซ็นเซอร์ล่าสุดที่ยังไม่ได้ประมวลผล
        # หาข้อมูลล่าสุดใน Model_Result ก่อน
        latest_processed = target_collection.find_one(sort=[("timestamp", -1)])
        
        # ตั้งค่าเงื่อนไขการค้นหา
        query = {}
        if latest_processed and 'timestamp' in latest_processed:
            # ดึงเฉพาะข้อมูลที่ใหม่กว่าข้อมูลล่าสุดที่ประมวลผลแล้ว
            query = {"timestamp": {"$gt": latest_processed['timestamp']}}
        
        # ดึงข้อมูลจาก source_collection
        cursor = source_collection.find(query).sort("timestamp", 1)
        
        # ประมวลผลแต่ละรายการ (ส่ง preprocessor ไปด้วย)
        count = 0
        for item in cursor:
            if process_one_document(item, model, model_columns, target_collection, preprocessor):
                count += 1
        
        logging.info(f"✅ ประมวลผลและบันทึกข้อมูลทั้งหมด {count} รายการ")
    
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        if client:
            client.close()

def run_realtime_processing():
    """ทำงานแบบ realtime โดยใช้ MongoDB Change Streams"""
    
    global running
    
    # โหลดโมเดล และ Preprocessor
    model, model_columns, preprocessor = load_model()
    if model is None or model_columns is None:
        logging.error("❌ ไม่สามารถโหลดโมเดลได้ ยกเลิกการประมวลผล")
        return
    
    # เชื่อมต่อกับ MongoDB
    client, source_collection, target_collection = connect_mongodb()
    if None in (client, source_collection, target_collection):
        logging.error("❌ ไม่สามารถเชื่อมต่อกับ MongoDB ได้ ยกเลิกการประมวลผล")
        return
    
    try:
        # ตรวจสอบจำนวนข้อมูลทั้งหมดในแต่ละ collection
        total_source = source_collection.count_documents({})
        total_target = target_collection.count_documents({})
        
        logging.info(f"📊 จำนวนข้อมูลทั้งหมด: sensordatas = {total_source}, Model_Result = {total_target}")
        
        # ถ้าจำนวนไม่เท่ากัน ต้องตรวจสอบและประมวลผลให้ครบ
        if total_source != total_target:
            logging.info(f"🔍 พบข้อมูลที่ยังไม่ได้ประมวลผล {total_source - total_target} รายการ")
            
            # ดึงรายการ source_id ที่มีการประมวลผลแล้ว
            processed_source_ids = set()
            target_cursor = target_collection.find({}, {"source_id": 1})
            for doc in target_cursor:
                if "source_id" in doc:
                    processed_source_ids.add(str(doc["source_id"]))
            
            logging.info(f"🔍 มีข้อมูลที่ประมวลผลแล้วทั้งหมด {len(processed_source_ids)} รายการ")
            
            # ค้นหาและประมวลผลข้อมูลที่ยังไม่ได้ประมวลผล
            count = 0
            source_cursor = source_collection.find({})
            for item in source_cursor:
                # ตรวจสอบตัวแปร running เพื่อให้สามารถหยุดได้ด้วย Ctrl+C
                if not running:
                    logging.info("⚠️ ได้รับคำสั่งหยุดการทำงาน ยกเลิกการประมวลผลข้อมูลที่เหลือ")
                    break
                    
                # ตรวจสอบว่า source_id นี้ได้รับการประมวลผลไปแล้วหรือไม่
                if str(item["_id"]) not in processed_source_ids:
                    if process_one_document(item, model, model_columns, target_collection, preprocessor):
                        count += 1
                        # อัพเดทเป็นระยะเพื่อแสดงความคืบหน้า
                        if count % 10 == 0:
                            logging.info(f"⏳ ประมวลผลข้อมูลไปแล้ว {count} รายการ...")
            
            logging.info(f"✅ ประมวลผลข้อมูลที่ตกหล่นเรียบร้อยแล้ว {count} รายการ")
            
            # ตรวจสอบอีกครั้งหลังจากประมวลผล
            total_target_after = target_collection.count_documents({})
            logging.info(f"📊 จำนวนข้อมูลหลังประมวลผล: sensordatas = {total_source}, Model_Result = {total_target_after}")
            
            if total_source != total_target_after:
                logging.warning(f"⚠️ จำนวนข้อมูลยังไม่เท่ากัน: ขาดอีก {total_source - total_target_after} รายการ")
        else:
            logging.info("✅ ข้อมูลมีความครบถ้วนแล้ว ทุกรายการได้รับการประมวลผล")
        
        # เริ่มการดักจับเหตุการณ์เพิ่มข้อมูลใหม่ด้วย Change Stream
        logging.info("⏱️ เริ่มการเฝ้าติดตามข้อมูลใหม่แบบ realtime...")
        
        # สร้าง pipeline สำหรับตรวจจับการเพิ่มข้อมูลใหม่
        pipeline = [{'$match': {'operationType': 'insert'}}]
        
        # สร้าง Change Stream แบบมี Resume Token เพื่อให้กลับมาทำงานต่อได้หากเกิดข้อผิดพลาด
        resume_token = None
        
        # เริ่มลูปหลักในการติดตามข้อมูลใหม่
        logging.info("✅ ระบบพร้อมแล้ว! กำลังรอข้อมูลใหม่... (กด Ctrl+C เพื่อออก)")
        
        while running:
            try:
                # เริ่มการเฝ้าดู change stream แบบมี timeout และมีการตรวจสอบสถานะ running เป็นระยะ
                options = {}
                if resume_token:
                    options['resume_after'] = resume_token
                
                with source_collection.watch(pipeline, **options) as stream:
                    # กำหนดเวลาหมดอายุให้กับ next() เพื่อไม่ให้บล็อกการทำงานนานเกินไป
                    # วนลูปเพื่อตรวจสอบสถานะ running เป็นระยะ
                    while running:
                        try:
                            # ใช้เทคนิค non-blocking ด้วย asyncio หรือ threading ไม่ได้
                            # จึงต้องใช้ timeout แทน
                            change = None
                            
                            # ดักจับข้อผิดพลาดแบบเจาะจงมากขึ้น
                            try:
                                # ใช้ get_more timeout ซึ่งเป็นคุณสมบัติของ MongoDB driver
                                # โดยกำหนดให้รอแค่ 2 วินาที แล้วกลับมาตรวจสอบสถานะ running
                                change = next(stream, None)
                                # เก็บ resume token เมื่อมีการเปลี่ยนแปลง
                                if change and '_id' in change:
                                    resume_token = change['_id']
                            except StopIteration:
                                # หมดเวลาในการรอข้อมูลใหม่
                                pass
                            
                            if change:
                                # ดึงเอกสารที่เพิ่มเข้ามาใหม่
                                new_document = change['fullDocument']
                                logging.info(f"🔔 มีข้อมูลใหม่เข้ามา: {new_document.get('_id')}")
                                
                                # ประมวลผลข้อมูลใหม่
                                if process_one_document(new_document, model, model_columns, target_collection, preprocessor):
                                    logging.info("✨ ประมวลผลข้อมูลใหม่เรียบร้อย!")
                                else:
                                    logging.warning("⚠️ ไม่สามารถประมวลผลข้อมูลใหม่ได้")
                            else:
                                # ไม่มีข้อมูลใหม่ รออีกสักครู่แล้วตรวจสอบสถานะ running อีกครั้ง
                                time.sleep(0.5)
                                
                        except Exception as inner_e:
                            logging.error(f"❌ เกิดข้อผิดพลาดในลูปการติดตาม: {str(inner_e)}")
                            import traceback
                            logging.error(traceback.format_exc())
                            # พักก่อนที่จะลองอีกครั้ง
                            time.sleep(2)
                            
                            # ตรวจสอบว่ายังต้องการทำงานต่อหรือไม่
                            if not running:
                                break
                                
                # หากออกจาก with block โดยไม่มีข้อผิดพลาด แสดงว่ามีการหยุดโปรแกรมอย่างปกติ
                # สามารถออกจากลูป while running ได้เลย
                if not running:
                    break
                    
            except Exception as e:
                logging.error(f"❌ เกิดข้อผิดพลาดในการเฝ้าติดตามข้อมูลใหม่: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                # รอสักครู่ก่อนลองเชื่อมต่อใหม่
                time.sleep(5)
                
                # ตรวจสอบว่ายังต้องการทำงานต่อหรือไม่
                if not running:
                    break
                
        logging.info("👋 หยุดการเฝ้าติดตามข้อมูลใหม่แล้ว")
                
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลแบบ realtime: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        # ตรวจสอบว่า client ยังอยู่หรือไม่
        if 'client' in locals() and client:
            try:
                client.close()
                logging.info("🔌 ปิดการเชื่อมต่อกับ MongoDB แล้ว")
            except Exception as e:
                logging.error(f"❌ เกิดข้อผิดพลาดในการปิดการเชื่อมต่อกับ MongoDB: {str(e)}")

if __name__ == "__main__":
    # แสดงข้อความต้อนรับ
    print("\n" + "="*60)
    print(" 🌟  ระบบประมวลผลคุณภาพอากาศจากข้อมูลเซ็นเซอร์  🌟")
    print("="*60)
    
    # ตรวจสอบพารามิเตอร์สั่งการ
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        print("\n🧠 กำลังประมวลผลข้อมูลแบบครั้งเดียวจาก MongoDB...")
        # ประมวลผลข้อมูลแบบเดิม (ครั้งเดียว)
        process_sensors_data()
    else:
        print("\n⏱️ กำลังเริ่มระบบประมวลผลแบบ Realtime...")
        print("   (สามารถกด Ctrl+C เพื่อหยุดการทำงาน)")
        # ประมวลผลข้อมูลแบบ realtime
        run_realtime_processing()
    
    print("\n" + "="*60)
    print("✅ การประมวลผลเสร็จสิ้น")
    print("="*60 + "\n") 