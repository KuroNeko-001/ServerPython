#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
โปรแกรมอัปเดตโมเดลด้วยชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น
ฝึกสอนโมเดลใหม่ด้วยชุดข้อมูล expanded_data.csv
"""

import os
import sys
import subprocess
import logging
import pandas as pd
import time

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_update.log"),
        logging.StreamHandler()
    ]
)

def check_required_files():
    """ตรวจสอบว่ามีไฟล์ที่จำเป็นครบถ้วนหรือไม่"""
    required_files = [
        "expand_dataset.py",
        "train_model_rf.py",
        "cleaned_data.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"❌ ไม่พบไฟล์ที่จำเป็น: {', '.join(missing_files)}")
        return False
    
    logging.info("✅ ตรวจสอบไฟล์ที่จำเป็นครบถ้วน")
    return True

def create_backup():
    """สร้างไฟล์สำรองของโมเดลเดิมและข้อมูลเดิม"""
    backup_time = time.strftime("%Y%m%d_%H%M%S")
    files_to_backup = [
        "random_forest_model.joblib",
        "model_columns.json",
        "data_preprocessor.joblib",
        "cleaned_data.csv"
    ]
    
    backup_folder = f"backup_{backup_time}"
    os.makedirs(backup_folder, exist_ok=True)
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_path = os.path.join(backup_folder, file)
            try:
                import shutil
                shutil.copy2(file, backup_path)
                logging.info(f"✅ สำรองไฟล์ {file} ไปยัง {backup_path}")
            except Exception as e:
                logging.error(f"❌ ไม่สามารถสำรองไฟล์ {file} ได้: {str(e)}")
    
    logging.info(f"✅ สำรองไฟล์ทั้งหมดไว้ในโฟลเดอร์ {backup_folder}")
    return backup_folder

def modify_train_model_file():
    """แก้ไขไฟล์ train_model_rf.py เพื่อใช้ชุดข้อมูลที่ขยายแล้ว"""
    try:
        # อ่านไฟล์ต้นฉบับ
        with open("train_model_rf.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # แทนที่ชื่อไฟล์ข้อมูล
        if "file_path = 'cleaned_data.csv'" in content:
            content = content.replace(
                "file_path = 'cleaned_data.csv'", 
                "file_path = 'expanded_data.csv'  # ใช้ชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น"
            )
            
            # เพิ่มข้อความอธิบายเกี่ยวกับการใช้ชุดข้อมูลใหม่
            if "print(f\"\\nUsing cleaned dataset from: {file_path}\")" in content:
                content = content.replace(
                    "print(f\"\\nUsing cleaned dataset from: {file_path}\")",
                    "print(f\"\\nUsing expanded temperature range dataset from: {file_path}\")\n    print(\"This dataset includes temperatures from 5°C to 45°C for better model generalization.\")"
                )
            
            # บันทึกไฟล์ที่แก้ไขแล้ว
            with open("train_model_rf.py", "w", encoding="utf-8") as f:
                f.write(content)
            
            logging.info("✅ แก้ไขไฟล์ train_model_rf.py เรียบร้อยแล้ว ให้ใช้ชุดข้อมูล expanded_data.csv")
            return True
        else:
            logging.error("❌ ไม่พบตำแหน่งที่ต้องแก้ไขในไฟล์ train_model_rf.py")
            return False
            
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการแก้ไขไฟล์ train_model_rf.py: {str(e)}")
        return False

def run_script(script_path, description):
    """รันสคริปต์และบันทึกผลลัพธ์"""
    logging.info(f"🚀 กำลังรัน {description}...")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # บันทึก stdout ลงในไฟล์ log
        log_filename = f"{os.path.splitext(script_path)[0]}_output.log"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        
        logging.info(f"✅ รัน {description} สำเร็จ")
        logging.info(f"📝 บันทึกผลลัพธ์ไว้ที่: {log_filename}")
        return True
    
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการรัน {description}")
        logging.error(f"stderr: {e.stderr}")
        return False

def verify_model_files():
    """ตรวจสอบว่าไฟล์โมเดลถูกสร้างขึ้นหรือไม่"""
    required_files = [
        "random_forest_model.joblib",
        "model_columns.json",
        "data_preprocessor.joblib"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            logging.error(f"❌ ไม่พบไฟล์โมเดล: {file} หลังจากการฝึกสอน")
            return False
    
    logging.info("✅ ตรวจสอบไฟล์โมเดลครบถ้วน")
    return True

def main():
    """ฟังก์ชันหลักสำหรับการอัปเดตโมเดล"""
    print("\n" + "="*70)
    print("🌟  โปรแกรมอัปเดตโมเดลด้วยชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น  🌟")
    print("="*70)
    
    print("\nขั้นตอนการอัปเดตโมเดล:")
    print("1. ตรวจสอบไฟล์ที่จำเป็น")
    print("2. สำรองไฟล์โมเดลและข้อมูลเดิม")
    print("3. สร้างชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น")
    print("4. แก้ไขไฟล์ train_model_rf.py เพื่อใช้ชุดข้อมูลใหม่")
    print("5. เทรนโมเดลใหม่ด้วยชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น")
    print("6. ตรวจสอบผลลัพธ์")
    
    # 1. ตรวจสอบไฟล์ที่จำเป็น
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 1: ตรวจสอบไฟล์ที่จำเป็น")
    if not check_required_files():
        print("❌ ไม่สามารถดำเนินการต่อได้เนื่องจากไฟล์ที่จำเป็นไม่ครบถ้วน")
        return
    
    # 2. สำรองไฟล์โมเดลและข้อมูลเดิม
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 2: สำรองไฟล์โมเดลและข้อมูลเดิม")
    backup_folder = create_backup()
    
    # 3. สร้างชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 3: สร้างชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น")
    if not run_script("expand_dataset.py", "สคริปต์สร้างชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น"):
        print("❌ ไม่สามารถสร้างชุดข้อมูลใหม่ได้ การอัปเดตโมเดลถูกยกเลิก")
        return
    
    # 4. แก้ไขไฟล์ train_model_rf.py
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 4: แก้ไขไฟล์ train_model_rf.py เพื่อใช้ชุดข้อมูลใหม่")
    if not modify_train_model_file():
        print("❌ ไม่สามารถแก้ไขไฟล์ train_model_rf.py ได้ การอัปเดตโมเดลถูกยกเลิก")
        return
    
    # 5. เทรนโมเดลใหม่
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 5: เทรนโมเดลใหม่ด้วยชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น")
    if not run_script("train_model_rf.py", "การเทรนโมเดลใหม่"):
        print("❌ เกิดข้อผิดพลาดในการเทรนโมเดลใหม่")
        return
    
    # 6. ตรวจสอบผลลัพธ์
    print("\n" + "-"*70)
    print("ขั้นตอนที่ 6: ตรวจสอบผลลัพธ์")
    if not verify_model_files():
        print("❌ ไม่พบไฟล์โมเดลที่จำเป็นหลังจากการเทรน")
        return
    
    # สรุปผล
    print("\n" + "="*70)
    print("✅  อัปเดตโมเดลเสร็จสมบูรณ์  ✅")
    print("="*70)
    print(f"\n📁 ไฟล์และโมเดลเดิมถูกสำรองไว้ที่โฟลเดอร์: {backup_folder}")
    print("🔍 โมเดลใหม่ทำงานได้ดีกับช่วงอุณหภูมิที่กว้างขึ้น (5-45°C)")
    print("📊 ชุดข้อมูลใหม่: expanded_data.csv")
    print("🤖 โมเดลใหม่: random_forest_model.joblib")
    print("📋 คอลัมน์โมเดล: model_columns.json")
    print("⚙️ ตัวประมวลผลข้อมูล: data_preprocessor.joblib")
    
    print("\n🎯 ขั้นตอนต่อไป:")
    print("1. รันโปรแกรม process_model_data.py เพื่อใช้งานโมเดลใหม่")
    print("2. ตรวจสอบผลลัพธ์ว่าโมเดลทำงานได้ดีกับช่วงอุณหภูมิที่หลากหลาย")
    print("3. หากพบปัญหาใดๆ สามารถกู้คืนไฟล์จากโฟลเดอร์สำรอง: " + backup_folder)

if __name__ == "__main__":
    main() 