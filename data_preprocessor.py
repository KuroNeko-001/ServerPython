"""
คลาส DataPreprocessor สำหรับการประมวลผลข้อมูลก่อนส่งเข้าโมเดล
แยกเป็นไฟล์ต่างหากเพื่อให้สามารถนำเข้าได้จากทั้ง train_model_rf.py และ process_model_data.py
"""

import pandas as pd
import numpy as np
import joblib

class DataPreprocessor:
    """
    คลาสนี้รับผิดชอบการเตรียมข้อมูลก่อนส่งเข้าโมเดล ทั้งในขั้นตอนเทรนและทำนาย
    เพื่อให้มั่นใจว่าการเตรียมข้อมูลเหมือนกันทั้งในขั้นตอนเทรนและใช้งานจริง
    """
    
    def __init__(self):
        # พารามิเตอร์สำหรับช่วงอุณหภูมิที่เหมาะสม
        self.temp_comfort_min = 22.0  # อุณหภูมิต่ำสุดที่สบาย (°C)
        self.temp_comfort_max = 28.0  # อุณหภูมิสูงสุดที่สบาย (°C)
        self.temp_hot_threshold = 35.0  # เริ่มร้อนเกินไป
        self.temp_very_hot_threshold = 38.0  # ร้อนมาก
        self.temp_extreme_hot_threshold = 40.0  # ร้อนรุนแรง
        self.temp_cold_threshold = 15.0  # เริ่มหนาวเกินไป
        self.temp_very_cold_threshold = 10.0  # หนาวมาก
    
    def create_temperature_features(self, data):
        """สร้างฟีเจอร์เกี่ยวกับอุณหภูมิ"""
        if 'temperature' not in data.columns:
            print("⚠️ Warning: 'temperature' column not found, skipping temperature features")
            return data
        
        # คัดลอกข้อมูลเพื่อไม่แก้ไขข้อมูลต้นฉบับ
        df = data.copy()
        
        # สร้างฟีเจอร์ความเบี่ยงเบนจากช่วงอุณหภูมิที่เหมาะสม
        df['temp_deviation'] = df['temperature'].apply(
            lambda x: max(0, x - self.temp_comfort_max) + max(0, self.temp_comfort_min - x)
        )
        
        # สร้างฟีเจอร์ว่าอุณหภูมิอยู่ในช่วงสบายหรือไม่ (0/1)
        df['temp_comfortable'] = df['temperature'].apply(
            lambda x: 1 if (self.temp_comfort_min <= x <= self.temp_comfort_max) else 0
        )
        
        # สร้างฟีเจอร์ว่าอุณหภูมิร้อนเกินไปหรือไม่ (0-3 โดย 0 = ปกติ, 3 = ร้อนมาก)
        df['temp_too_hot'] = df['temperature'].apply(
            lambda x: 3 if x >= self.temp_extreme_hot_threshold else 
                      (2 if x >= self.temp_very_hot_threshold else 
                       (1 if x >= self.temp_hot_threshold else 0))
        )
        
        # สร้างฟีเจอร์ว่าอุณหภูมิหนาวเกินไปหรือไม่ (0-2 โดย 0 = ปกติ, 2 = หนาวมาก)
        df['temp_too_cold'] = df['temperature'].apply(
            lambda x: 2 if x <= self.temp_very_cold_threshold else 
                      (1 if x <= self.temp_cold_threshold else 0)
        )
        
        # ผลรวมของความผิดปกติด้านอุณหภูมิ
        df['temp_abnormality'] = df['temp_too_hot'] + df['temp_too_cold']
        
        print(f"✅ เพิ่มฟีเจอร์อุณหภูมิแล้ว: temp_deviation, temp_comfortable, temp_too_hot, temp_too_cold, temp_abnormality")
        
        return df
    
    def create_humidity_features(self, data):
        """สร้างฟีเจอร์เกี่ยวกับความชื้น"""
        if 'humidity' not in data.columns:
            print("⚠️ Warning: 'humidity' column not found, skipping humidity features")
            return data
        
        # คัดลอกข้อมูลเพื่อไม่แก้ไขข้อมูลต้นฉบับ
        df = data.copy()
        
        # สร้างฟีเจอร์ความเบี่ยงเบนจากช่วงความชื้นที่เหมาะสม (40-70%)
        df['humid_deviation'] = df['humidity'].apply(
            lambda x: max(0, x - 70) + max(0, 40 - x)
        )
        
        # สร้างฟีเจอร์ว่าความชื้นสูงเกินไปหรือไม่ (0/1)
        df['humid_too_high'] = (df['humidity'] > 70).astype(int)
        
        # สร้างฟีเจอร์ว่าความชื้นต่ำเกินไปหรือไม่ (0/1)
        df['humid_too_low'] = (df['humidity'] < 40).astype(int)
        
        print(f"✅ เพิ่มฟีเจอร์ความชื้นแล้ว: humid_deviation, humid_too_high, humid_too_low")
        
        return df
    
    def create_thermal_comfort_features(self, data):
        """สร้างฟีเจอร์ความสบายด้านความร้อน (Thermal Comfort)"""
        if 'temperature' not in data.columns or 'humidity' not in data.columns:
            print("⚠️ Warning: 'temperature' or 'humidity' column not found, skipping thermal comfort features")
            return data
        
        # คัดลอกข้อมูลเพื่อไม่แก้ไขข้อมูลต้นฉบับ
        df = data.copy()
        
        # Heat Index อย่างง่าย (ดัชนีความร้อน)
        # ผลกระทบของความชื้นต่อความรู้สึกร้อน
        df['heat_index'] = df.apply(
            lambda row: row['temperature'] * (1 + 0.05 * max(0, row['humidity'] - 50)), axis=1
        )
        
        # สร้างฟีเจอร์ความไม่สบายด้านความร้อน (0-5)
        df['thermal_discomfort'] = df.apply(
            lambda row: 
                5 if row['heat_index'] >= 45 else  # อันตรายถึงชีวิต
                4 if row['heat_index'] >= 40 else  # เครียดจากความร้อนรุนแรง
                3 if row['heat_index'] >= 35 else  # เครียดจากความร้อน
                2 if row['heat_index'] >= 32 else  # ระวัง
                1 if row['heat_index'] >= 30 else  # ควรระวัง
                0,  # สบาย
            axis=1
        )
        
        print(f"✅ เพิ่มฟีเจอร์ความสบายด้านความร้อนแล้ว: heat_index, thermal_discomfort")
        
        return df
    
    def process_data(self, data):
        """ประมวลผลข้อมูลทั้งหมด"""
        df = data.copy()
        
        # เพิ่มฟีเจอร์ต่างๆ
        df = self.create_temperature_features(df)
        df = self.create_humidity_features(df)
        df = self.create_thermal_comfort_features(df)
        
        return df
    
    def save(self, filename="data_preprocessor.joblib"):
        """บันทึกตัวประมวลผลข้อมูลเพื่อนำไปใช้ในการทำนาย"""
        joblib.dump(self, filename)
        print(f"✅ บันทึกตัวประมวลผลข้อมูลไปยัง {filename}")
        return filename 