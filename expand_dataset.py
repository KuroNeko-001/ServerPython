import pandas as pd
import numpy as np
import os
import logging

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_expansion.log"),
        logging.StreamHandler()
    ]
)

def expand_temperature_range(input_file="cleaned_data.csv", output_file="expanded_data.csv"):
    """
    สร้างชุดข้อมูลใหม่ที่มีช่วงอุณหภูมิกว้างกว่าเดิม
    โดยเพิ่มข้อมูลอุณหภูมิสูง (35-45°C) และอุณหภูมิต่ำ (5-15°C)
    """
    try:
        # ตรวจสอบว่าไฟล์นำเข้ามีอยู่หรือไม่
        if not os.path.exists(input_file):
            logging.error(f"❌ ไม่พบไฟล์ข้อมูล: {input_file}")
            return False
            
        # โหลดชุดข้อมูลเดิม
        logging.info(f"📂 กำลังโหลดชุดข้อมูลจาก {input_file}...")
        data = pd.read_csv(input_file)
        
        # แสดงข้อมูลเบื้องต้นของชุดข้อมูลเดิม
        logging.info(f"📊 ชุดข้อมูลเดิมมี {len(data)} แถว")
        logging.info(f"🌡️ ช่วงอุณหภูมิเดิม: {data['temperature'].min():.1f}-{data['temperature'].max():.1f}°C")
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['temperature', 'humidity', 'aqi_class', 'pm2_5', 'pm10']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
            return False
        
        # 1. สร้างชุดข้อมูลอุณหภูมิสูง (35-45°C)
        logging.info("🔥 กำลังสร้างชุดข้อมูลอุณหภูมิสูง (35-45°C)...")
        high_temp_samples = []
        
        for temp in np.linspace(35, 45, 100):  # 100 ตัวอย่างจาก 35-45°C
            # คัดลอกข้อมูลจากชุดเดิม 10 แถว
            for i in range(10):
                sample_idx = np.random.randint(0, len(data))
                new_sample = data.iloc[sample_idx].copy()
                
                # ปรับอุณหภูมิและความชื้น
                new_sample['temperature'] = temp
                new_sample['humidity'] = min(new_sample['humidity'], 70)  # ความชื้นไม่ควรสูงเกินไปเมื่ออุณหภูมิสูง
                
                # ปรับค่า PM2.5 และ PM10 เพิ่มขึ้นเล็กน้อยเมื่ออุณหภูมิสูง (เนื่องจากอุณหภูมิสูงมักทำให้ฝุ่นลอยตัวได้ดีขึ้น)
                humidity_factor = new_sample['humidity'] / 100.0  # ความชื้นสูงช่วยลดฝุ่น
                temp_factor = (temp - 35) / 10.0  # อุณหภูมิยิ่งสูงยิ่งเพิ่มฝุ่น
                
                # ปรับค่า PM2.5 และ PM10 ตามปัจจัยอุณหภูมิและความชื้น
                if 'pm2_5' in new_sample:
                    new_sample['pm2_5'] = new_sample['pm2_5'] * (1 + 0.2 * temp_factor - 0.1 * humidity_factor)
                    new_sample['pm2_5'] = max(5.0, min(150.0, new_sample['pm2_5']))  # จำกัดขอบเขต
                
                if 'pm10' in new_sample:
                    new_sample['pm10'] = new_sample['pm10'] * (1 + 0.15 * temp_factor - 0.08 * humidity_factor)
                    new_sample['pm10'] = max(10.0, min(250.0, new_sample['pm10']))  # จำกัดขอบเขต
                
                # ปรับค่า AQI class ตามอุณหภูมิ
                if temp >= 40:
                    new_sample['aqi_class'] = 4
                elif temp >= 38:
                    new_sample['aqi_class'] = max(new_sample['aqi_class'], 3)
                elif temp >= 35:
                    new_sample['aqi_class'] = max(new_sample['aqi_class'], 2)
                
                high_temp_samples.append(new_sample)
        
        logging.info(f"✅ สร้างข้อมูลอุณหภูมิสูงเสร็จสิ้น: {len(high_temp_samples)} ตัวอย่าง")
        
        # 2. สร้างชุดข้อมูลอุณหภูมิต่ำ (5-15°C)
        logging.info("❄️ กำลังสร้างชุดข้อมูลอุณหภูมิต่ำ (5-15°C)...")
        low_temp_samples = []
        
        for temp in np.linspace(5, 15, 50):  # 50 ตัวอย่างจาก 5-15°C
            # คัดลอกข้อมูลจากชุดเดิม 10 แถว
            for i in range(10):
                sample_idx = np.random.randint(0, len(data))
                new_sample = data.iloc[sample_idx].copy()
                
                # ปรับอุณหภูมิและความชื้น
                new_sample['temperature'] = temp
                new_sample['humidity'] = min(max(new_sample['humidity'], 30), 85)  # ปรับความชื้นให้สมจริงในอุณหภูมิต่ำ
                
                # ปรับค่า AQI class ตามอุณหภูมิ
                if temp <= 10:
                    new_sample['aqi_class'] = max(new_sample['aqi_class'], 3)
                elif temp <= 15:
                    new_sample['aqi_class'] = max(new_sample['aqi_class'], 2)
                
                low_temp_samples.append(new_sample)
        
        logging.info(f"✅ สร้างข้อมูลอุณหภูมิต่ำเสร็จสิ้น: {len(low_temp_samples)} ตัวอย่าง")
        
        # 3. สร้างชุดข้อมูลอุณหภูมิกลาง (เพิ่มเติมจาก 16-34°C) เพื่อให้มีข้อมูลสม่ำเสมอ
        logging.info("🌡️ กำลังสร้างชุดข้อมูลอุณหภูมิกลางเพิ่มเติม (16-34°C)...")
        mid_temp_samples = []
        
        for temp in np.linspace(16, 34, 60):  # 60 ตัวอย่างจาก 16-34°C
            # คัดลอกข้อมูลจากชุดเดิม 5 แถว
            for i in range(5):
                sample_idx = np.random.randint(0, len(data))
                new_sample = data.iloc[sample_idx].copy()
                
                # ปรับอุณหภูมิ
                new_sample['temperature'] = temp
                
                # ปรับค่า AQI class และความชื้นตามปัจจัย
                if 28 <= temp <= 34:
                    if new_sample['humidity'] > 75:
                        # อุณหภูมิสูงและชื้นสูง = รู้สึกไม่สบาย
                        new_sample['aqi_class'] = max(new_sample['aqi_class'], 2)
                
                mid_temp_samples.append(new_sample)
        
        logging.info(f"✅ สร้างข้อมูลอุณหภูมิกลางเสร็จสิ้น: {len(mid_temp_samples)} ตัวอย่าง")
        
        # 4. รวมชุดข้อมูลเดิมกับชุดข้อมูลใหม่
        high_temp_df = pd.DataFrame(high_temp_samples)
        low_temp_df = pd.DataFrame(low_temp_samples)
        mid_temp_df = pd.DataFrame(mid_temp_samples)
        
        combined_df = pd.concat([data, high_temp_df, low_temp_df, mid_temp_df], ignore_index=True)
        
        # 5. ตรวจสอบและแก้ไขข้อมูลที่ไม่ถูกต้อง
        logging.info("🔍 กำลังตรวจสอบและแก้ไขข้อมูลที่ไม่ถูกต้อง...")
        
        # ตรวจสอบว่าค่า AQI class อยู่ในช่วงที่ถูกต้อง (0-4)
        combined_df['aqi_class'] = combined_df['aqi_class'].clip(0, 4)
        
        # ตรวจสอบและซ่อมแซมค่า NaN หรือค่าผิดปกติ
        for col in combined_df.columns:
            # ตรวจสอบและซ่อมแซมค่า NaN
            if combined_df[col].isna().any():
                if col in ['temperature', 'humidity', 'pm2_5', 'pm10']:
                    # สำหรับคอลัมน์ตัวเลขสำคัญใช้ค่าเฉลี่ย
                    mean_value = combined_df[col].mean()
                    combined_df[col].fillna(mean_value, inplace=True)
                    logging.info(f"🔧 แก้ไขค่า NaN ในคอลัมน์ {col} ด้วยค่าเฉลี่ย: {mean_value:.2f}")
                else:
                    # สำหรับคอลัมน์อื่นๆ ใช้ค่าที่พบบ่อยที่สุด
                    mode_value = combined_df[col].mode()[0]
                    combined_df[col].fillna(mode_value, inplace=True)
                    logging.info(f"🔧 แก้ไขค่า NaN ในคอลัมน์ {col} ด้วยค่าที่พบบ่อยที่สุด: {mode_value}")
        
        # 6. บันทึกชุดข้อมูลใหม่
        combined_df.to_csv(output_file, index=False)
        
        # 7. แสดงสรุปข้อมูล
        logging.info(f"✅ สร้างชุดข้อมูลใหม่ที่มีอุณหภูมิหลากหลายแล้ว: {len(combined_df)} แถว")
        logging.info(f"🌡️ ช่วงอุณหภูมิใหม่: {combined_df['temperature'].min():.1f}-{combined_df['temperature'].max():.1f}°C")
        
        # สรุปการกระจายของ AQI class
        aqi_counts = combined_df['aqi_class'].value_counts().sort_index()
        logging.info(f"📊 การกระจายของ AQI class:")
        for aqi_class, count in aqi_counts.items():
            percentage = (count / len(combined_df)) * 100
            logging.info(f"   ระดับ {aqi_class}: {count} ตัวอย่าง ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        import traceback
        logging.error(f"❌ เกิดข้อผิดพลาดในการสร้างชุดข้อมูล: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def main():
    """ฟังก์ชันหลักของโปรแกรม"""
    print("\n" + "="*70)
    print("🌟  โปรแกรมสร้างชุดข้อมูลคุณภาพอากาศที่มีช่วงอุณหภูมิกว้างขึ้น  🌟")
    print("="*70 + "\n")
    
    # ขยายช่วงอุณหภูมิของชุดข้อมูล
    success = expand_temperature_range()
    
    if success:
        print("\n" + "="*70)
        print("✅  การสร้างชุดข้อมูลใหม่เสร็จสมบูรณ์  ✅")
        print("="*70)
        print("\n🔍 ตรวจสอบไฟล์ expanded_data.csv สำหรับชุดข้อมูลที่ขยายช่วงอุณหภูมิแล้ว")
        print("📝 ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ dataset_expansion.log")
    else:
        print("\n" + "="*70)
        print("❌  เกิดข้อผิดพลาดในการสร้างชุดข้อมูลใหม่  ❌")
        print("="*70)
        print("\n⚠️ กรุณาตรวจสอบไฟล์ dataset_expansion.log สำหรับรายละเอียดข้อผิดพลาด")
    
if __name__ == "__main__":
    main() 