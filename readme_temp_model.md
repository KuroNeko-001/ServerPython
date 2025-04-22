# วิธีการปรับปรุงโมเดลให้รองรับอุณหภูมิที่หลากหลาย

ปัญหาที่พบในโมเดลปัจจุบัน:
**ชุดข้อมูลที่ใช้ฝึกสอนมีอุณหภูมิอยู่ในช่วงแคบเกินไป (20.5-27.7°C)** ทำให้โมเดลไม่สามารถจัดการกับค่าอุณหภูมิที่สูงมากหรือต่ำมากได้อย่างเหมาะสม

## วิธีแก้ไขระยะสั้น (ที่ได้ทำแล้ว):
1. เพิ่มโค้ดในไฟล์ `process_model_data.py` เพื่อตรวจสอบและปรับระดับคุณภาพอากาศหลังจากการทำนายของโมเดล
2. ถ้าอุณหภูมิ ≥ 40°C จะปรับเป็นระดับ 4 (แย่ที่สุด)
3. ถ้าอุณหภูมิ ≥ 38°C จะปรับเป็นอย่างน้อยระดับ 3
4. ถ้าอุณหภูมิ ≥ 35°C จะปรับเป็นอย่างน้อยระดับ 2
5. ใช้ความสบายด้านความร้อน (thermal_discomfort) เพื่อปรับปรุงเพิ่มเติม

## วิธีแก้ไขระยะยาว (ควรทำในอนาคต):
1. สร้างชุดข้อมูลใหม่ที่มีอุณหภูมิหลากหลายกว่าเดิม
2. เทรนโมเดลใหม่ด้วยชุดข้อมูลนี้

## วิธีการสร้างชุดข้อมูลที่มีอุณหภูมิหลากหลาย:

### 1. เพิ่มข้อมูลอุณหภูมิสูงและต่ำลงในชุดข้อมูลเดิม

```python
import pandas as pd
import numpy as np

# โหลดชุดข้อมูลเดิม
data = pd.read_csv("cleaned_data.csv")

# สร้างชุดข้อมูลอุณหภูมิสูง (35-45°C)
high_temp_samples = []
for temp in np.linspace(35, 45, 100):  # 100 ตัวอย่างจาก 35-45°C
    # คัดลอกข้อมูลจากชุดเดิม 10 แถว
    for i in range(10):
        sample_idx = np.random.randint(0, len(data))
        new_sample = data.iloc[sample_idx].copy()
        # ปรับอุณหภูมิและ heat index
        new_sample['temperature'] = temp
        new_sample['humidity'] = min(new_sample['humidity'], 70)  # ความชื้นไม่ควรสูงเกินไปเมื่ออุณหภูมิสูง
        
        # ปรับค่า AQI class ตามอุณหภูมิ
        if temp >= 40:
            new_sample['aqi_class'] = 4
        elif temp >= 38:
            new_sample['aqi_class'] = max(new_sample['aqi_class'], 3)
        elif temp >= 35:
            new_sample['aqi_class'] = max(new_sample['aqi_class'], 2)
            
        high_temp_samples.append(new_sample)

# สร้างชุดข้อมูลอุณหภูมิต่ำ (5-15°C)
low_temp_samples = []
for temp in np.linspace(5, 15, 50):  # 50 ตัวอย่างจาก 5-15°C
    # คัดลอกข้อมูลจากชุดเดิม 10 แถว
    for i in range(10):
        sample_idx = np.random.randint(0, len(data))
        new_sample = data.iloc[sample_idx].copy()
        # ปรับอุณหภูมิ
        new_sample['temperature'] = temp
        
        # ปรับค่า AQI class
        if temp <= 10:
            new_sample['aqi_class'] = max(new_sample['aqi_class'], 3)
        elif temp <= 15:
            new_sample['aqi_class'] = max(new_sample['aqi_class'], 2)
            
        low_temp_samples.append(new_sample)

# รวมชุดข้อมูลเดิมกับชุดข้อมูลใหม่
high_temp_df = pd.DataFrame(high_temp_samples)
low_temp_df = pd.DataFrame(low_temp_samples)
combined_df = pd.concat([data, high_temp_df, low_temp_df], ignore_index=True)

# บันทึกชุดข้อมูลใหม่
combined_df.to_csv("expanded_data.csv", index=False)
print(f"สร้างชุดข้อมูลใหม่ที่มีอุณหภูมิหลากหลายแล้ว: {len(combined_df)} แถว")
print(f"ช่วงอุณหภูมิ: {combined_df['temperature'].min()}-{combined_df['temperature'].max()}°C")
```

### 2. ใช้ชุดข้อมูลใหม่เทรนโมเดล

1. เปลี่ยนชื่อไฟล์ในฟังก์ชัน `main()` ของไฟล์ `train_model_rf.py`:

```python
def main():
    # ...
    # ใช้ชุดข้อมูลใหม่แทน
    file_path = 'expanded_data.csv'  # แทนที่ cleaned_data.csv
    # ...
```

2. รันโค้ดเพื่อเทรนโมเดลใหม่:

```bash
python train_model_rf.py
```

3. ทดสอบด้วยข้อมูลที่มีอุณหภูมิสูงหรือต่ำเพื่อตรวจสอบว่าโมเดลทำงานได้ถูกต้อง

## สรุป:
1. แก้ไขระยะสั้น: โค้ดถูกแก้ไขให้จัดการกับอุณหภูมิสูงและต่ำแบบ post-processing
2. แก้ไขระยะยาว: ควรสร้างชุดข้อมูลที่มีอุณหภูมิหลากหลายและเทรนโมเดลใหม่

หมายเหตุ: กระบวนการนี้จะทำให้โมเดลสามารถรองรับอุณหภูมิในช่วงกว้างได้ดีขึ้นโดยไม่ต้องพึ่งการแก้ไขหลังการทำนายเท่านั้น 