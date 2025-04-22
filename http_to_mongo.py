from flask import Flask, request, jsonify
import json
import logging
import os
import datetime
import joblib
import pandas as pd
import time
from pymongo import MongoClient
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("air_quality_system.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# ===== Settings =====
# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "air_quality_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "sensor_readings")

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "random_forest_model.joblib")
MODEL_COLUMNS_PATH = os.getenv("MODEL_COLUMNS_PATH", "model_columns.json")

# ===== Load model and columns used for prediction =====
def load_model():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_COLUMNS_PATH):
            logging.error(f"❌ Model or columns file not found. Please run train_model_rf.py first")
            return None, None
            
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Load columns
        with open(MODEL_COLUMNS_PATH, 'r') as f:
            model_columns = json.load(f)
            
        logging.info("✅ Model loaded successfully")
        return model, model_columns
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return None, None

# ===== Connect to MongoDB =====
def connect_mongodb():
    try:
        # Connect to MongoDB Atlas
        logging.info(f"Connecting to MongoDB Atlas...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        
        # Check if collection exists, create if not
        if MONGO_COLLECTION not in db.list_collection_names():
            logging.info(f"Creating new collection: {MONGO_COLLECTION}")
            # Create collection with timestamp index
            db.create_collection(MONGO_COLLECTION)
            db[MONGO_COLLECTION].create_index([("timestamp", 1)])
        
        logging.info(f"✅ Connected to MongoDB Atlas successfully (Database: {MONGO_DB}, Collection: {MONGO_COLLECTION})")
        return collection
    except Exception as e:
        logging.error(f"❌ Error connecting to MongoDB: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# ===== Function to predict air quality from sensor data =====
def predict_air_quality(data, model, model_columns):
    try:
        # Add timestamp to data
        data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Check if PM2.5 value exists
        has_pm25 = 'pm2_5' in data and data['pm2_5'] is not None
        pm25_value = data.get('pm2_5', 0) if has_pm25 else 0
        
        # 1. Determine air quality level directly from PM2.5 first (according to Thailand standards)
        # Main decision will be based on PM2.5 if available
        if has_pm25:
            if pm25_value <= 25:
                direct_prediction = 0
            elif pm25_value <= 37:
                direct_prediction = 1
            elif pm25_value <= 50:
                direct_prediction = 2
            elif pm25_value <= 90:
                direct_prediction = 3
            else:
                direct_prediction = 4
        else:
            direct_prediction = None
            
        # 2. Create DataFrame for ML model prediction
        input_df = pd.DataFrame([data])
        
        # Estimate missing values (if necessary)
        missing_columns = []
        for col in model_columns:
            if col not in input_df.columns:
                missing_columns.append(col)
                
                # Intelligently estimate missing values
                if col == 'pm10' and 'pm2_5' in input_df.columns:
                    # PM10 is typically about 1.8-2.2 times PM2.5
                    input_df[col] = input_df['pm2_5'] * 1.85
                elif col == 'co' and 'co2' in input_df.columns:
                    # CO has relationship with CO2 (approximate)
                    input_df[col] = input_df['co2'] * 0.015
                elif col == 'o3' and 'temperature' in input_df.columns and 'humidity' in input_df.columns:
                    # O3 has relationship with temperature and humidity (approximate)
                    temp_factor = (input_df['temperature'] - 20) / 15  # normalized temperature effect
                    humid_factor = (70 - input_df['humidity']) / 50  # humidity inverse correlation
                    input_df[col] = 0.05 + (temp_factor * 0.05) + (humid_factor * 0.03)
                    input_df[col] = input_df[col].clip(0.01, 0.2)  # limit to reasonable range
                elif col == 'no2' and 'co2' in input_df.columns:
                    # NO2 has relationship with CO2 (approximate)
                    input_df[col] = 0.05 + (input_df['co2'] / 5000) * 0.2
                    input_df[col] = input_df[col].clip(0.01, 0.5)
                elif col == 'so2':
                    # SO2 is typically low indoors without direct sources
                    input_df[col] = 0.02
                elif col == 'pm2_5':
                    if 'pm10' in input_df.columns:
                        input_df[col] = input_df['pm10'] / 1.85  # estimate from PM10
                    else:
                        input_df[col] = 20.0  # average value
                else:
                    # For other columns, use reasonable default values
                    if col == 'temperature':
                        input_df[col] = 25.0  # average room temperature
                    elif col == 'humidity':
                        input_df[col] = 60.0  # average room humidity
                    elif col == 'co2':
                        input_df[col] = 800.0  # average in typical room
                    else:
                        input_df[col] = 0.0  # default for other columns
                
        # Show warning if data is missing
        if missing_columns:
            missing_msg = f"⚠️ Some data is missing and has been automatically estimated: {', '.join(missing_columns)}"
            logging.warning(missing_msg)
        
        # Arrange columns to match the model
        input_df = input_df[model_columns]
        
        # ========================= Prediction Steps =========================
        # 3. Predict using ML model (used as supplementary information)
        model_prediction = None
        prediction_method = "direct"  # Default, use direct (from PM2.5 directly)
        probabilities = None
        
        try:
            model_prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
        except Exception as e:
            logging.error(f"❌ Error in model prediction: {str(e)}")
            model_prediction = None
            # Will use prediction from PM2.5 directly instead
        
        # 4. Decide which prediction method to use
        if has_pm25:
            # If PM2.5 exists - give weight to direct PM2.5 value
            final_prediction = direct_prediction
            prediction_method = "direct"
            
            if model_prediction is not None:
                # Check difference between prediction methods
                if model_prediction != direct_prediction:
                    # If model prediction and PM2.5 differ by only 1 level
                    if abs(model_prediction - direct_prediction) == 1:
                        # Additional consideration: if PM2.5 is near level boundaries, consider both sides
                        if (pm25_value >= 23 and pm25_value <= 25) or \
                           (pm25_value >= 35 and pm25_value <= 37) or \
                           (pm25_value >= 48 and pm25_value <= 50) or \
                           (pm25_value >= 88 and pm25_value <= 90):
                            # PM2.5 near boundaries, give 30% importance to model
                            # (because model considers other factors too)
                            prediction_method = "hybrid_edge"
                            # 70/30 range: 70% PM2.5, 30% ML model
                            # Near upper boundary, if model gives higher value, use higher
                            if (pm25_value >= 23 and model_prediction > direct_prediction) or \
                               (pm25_value >= 35 and model_prediction > direct_prediction) or \
                               (pm25_value >= 48 and model_prediction > direct_prediction) or \
                               (pm25_value >= 88 and model_prediction > direct_prediction):
                                final_prediction = model_prediction
                            # Near lower boundary, if model gives lower value, use lower
                            elif (pm25_value <= 27 and model_prediction < direct_prediction) or \
                                 (pm25_value <= 39 and model_prediction < direct_prediction) or \
                                 (pm25_value <= 52 and model_prediction < direct_prediction) or \
                                 (pm25_value <= 93 and model_prediction < direct_prediction):
                                final_prediction = model_prediction
                    else:
                        # Difference more than 1 level, use PM2.5 as main factor
                        prediction_method = "pm25_override"
                        logging.warning(f"⚠️ Model prediction ({model_prediction}) differs significantly from PM2.5 ({direct_prediction})")
                        logging.warning(f"⚠️ Using PM2.5 = {pm25_value} μg/m³ as main decision factor → Level {direct_prediction}")
                else:
                    # Model and PM2.5 give same result - very high confidence
                    prediction_method = "consistent"
        else:
            # If no PM2.5 - must use ML model
            if model_prediction is not None:
                final_prediction = model_prediction
                prediction_method = "model"
            else:
                # No PM2.5 and model has issues, use average (moderate level)
                final_prediction = 2 
                prediction_method = "fallback"
                logging.warning("⚠️ No PM2.5 data and model has issues, using moderate level instead (Level 2)")
        
        # 5. Air quality levels according to Thailand standards
        aqi_labels = {
            0: "คุณภาพดีมาก (0-25 μg/m³)",
            1: "คุณภาพดี (26-37 μg/m³)",
            2: "ปานกลาง (38-50 μg/m³)",
            3: "เริ่มมีผลกระทบต่อสุขภาพ (51-90 μg/m³)",
            4: "มีผลกระทบต่อสุขภาพ (91+ μg/m³)"
        }
        
        # Add health risk information for each level (customized for classroom)
        health_risks = {
            0: "ความเสี่ยงต่ำมาก: คุณภาพอากาศในห้องเรียนยอดเยี่ยม เอื้อต่อการเรียนรู้และพัฒนาการของนักเรียน ไม่มีผลกระทบต่อสมาธิหรือประสิทธิภาพการเรียนรู้",
            1: "ความเสี่ยงต่ำ: คุณภาพอากาศดี ส่งผลกระทบน้อยมากต่อนักเรียนที่มีภูมิแพ้หรือมีปัญหาระบบทางเดินหายใจ ยังคงเหมาะสำหรับกิจกรรมการเรียนการสอนปกติ",
            2: "ความเสี่ยงปานกลาง: อาจทำให้เกิดการระคายเคืองตา จมูก และลำคอสำหรับนักเรียนบางคน ซึ่งอาจลดความสามารถในการมีสมาธิเล็กน้อยระหว่างเรียน",
            3: "ความเสี่ยงสูง: นักเรียนอาจมีช่วงความสนใจลดลง เหนื่อยล้า หรือปวดศีรษะเนื่องจากคุณภาพอากาศ ส่งผลกระทบโดยตรงต่อประสิทธิภาพการเรียนรู้",
            4: "ความเสี่ยงสูงมาก: นักเรียนส่วนใหญ่จะได้รับผลกระทบอย่างรุนแรงด้วยอาการเหนื่อยล้า หายใจลำบาก ไม่สามารถมีสมาธิ และอาจจำเป็นต้องระงับการเรียนการสอน"
        }
        
        # Add recommendations for each level (customized for classroom)
        health_recommendations = {
            0: "ดำเนินกิจกรรมในห้องเรียนได้ตามปกติ สภาพแวดล้อมเหมาะสมที่สุดสำหรับการเรียนการสอนทุกรูปแบบ",
            1: "เฝ้าสังเกตนักเรียนที่มีภูมิแพ้หรือมีปัญหาระบบทางเดินหายใจ สามารถดำเนินกิจกรรมในห้องเรียนได้ตามปกติ",
            2: "พิจารณาใช้เครื่องฟอกอากาศในห้องเรียน ลดกิจกรรมที่อาจสร้างฝุ่น และหลีกเลี่ยงการเปิดหน้าต่างเมื่อคุณภาพอากาศภายนอกไม่ดี",
            3: "ใช้เครื่องฟอกอากาศอย่างต่อเนื่อง พิจารณาลดกิจกรรมทางกาย เฝ้าสังเกตนักเรียนที่มีปัญหาสุขภาพ และอาจลดระยะเวลาเรียนหรือเพิ่มเวลาพัก",
            4: "พิจารณาหยุดการเรียนการสอนในห้องนี้หรือย้ายไปยังพื้นที่ที่มีคุณภาพอากาศดีกว่า นักเรียนควรสวมหน้ากาก N95 ในห้องเรียน ใช้เครื่องฟอกอากาศหลายเครื่องและเฝ้าสังเกตนักเรียนที่มีปัญหาสุขภาพอย่างใกล้ชิด"
        }
        
        # 6. Adjust probabilities to match prediction method
        if probabilities is None or len(probabilities) < 5:
            # Create new probabilities
            adjusted_probs = [0.0] * 5
            
            if prediction_method == "direct" or prediction_method == "pm25_override":
                # If decision is from PM2.5 directly, high confidence
                adjusted_probs[final_prediction] = 0.9
                
                # Distribute probability to nearby classes
                if final_prediction > 0:
                    adjusted_probs[final_prediction-1] = 0.05
                if final_prediction < 4:
                    adjusted_probs[final_prediction+1] = 0.05
                    
            elif prediction_method == "hybrid_edge":
                # Case using both PM2.5 and model
                adjusted_probs[direct_prediction] = 0.6
                adjusted_probs[model_prediction] = 0.3
                
                # Distribute remaining probability
                indices = [i for i in range(5) if i != direct_prediction and i != model_prediction]
                for idx in indices:
                    adjusted_probs[idx] = 0.1 / len(indices) if len(indices) > 0 else 0
                    
            elif prediction_method == "fallback":
                # Case with insufficient data, low confidence
                adjusted_probs[2] = 0.5  # moderate
                adjusted_probs[1] = 0.3  # good
                adjusted_probs[3] = 0.2  # begins to affect health
                
            probabilities = adjusted_probs
        elif prediction_method == "direct" and has_pm25:
            # Adjust probabilities to reflect higher weight on PM2.5
            adjusted_probs = list(probabilities)
            
            # Increase confidence for level matching PM2.5
            max_value = max(adjusted_probs)
            
            if adjusted_probs[direct_prediction] < 0.7:
                # Increase value to match direct prediction from PM2.5
                amount_to_add = 0.7 - adjusted_probs[direct_prediction]
                adjusted_probs[direct_prediction] += amount_to_add
                
                # Reduce values from other classes proportionally
                sum_others = sum(adjusted_probs) - adjusted_probs[direct_prediction]
                if sum_others > 0:
                    for i in range(len(adjusted_probs)):
                        if i != direct_prediction:
                            adjusted_probs[i] -= (amount_to_add * adjusted_probs[i] / sum_others)
            
            # Adjust sum to 1
            sum_probs = sum(adjusted_probs)
            if sum_probs > 0:
                adjusted_probs = [p/sum_probs for p in adjusted_probs]
                
            probabilities = adjusted_probs
        
        # 7. Create result
        result = {
            "aqi_class": int(final_prediction),
            "aqi_label": aqi_labels[final_prediction],
            "health_risk": health_risks[final_prediction],
            "health_recommendation": health_recommendations[final_prediction],
            "probabilities": {f"class_{i}": float(prob) for i, prob in enumerate(probabilities)},
            "method": prediction_method,
            "pm25_value": float(pm25_value) if has_pm25 else None
        }
        
        # 8. Add important sensor information to result
        key_sensors = ['temperature', 'humidity', 'co2']
        sensor_info = {}
        for sensor in key_sensors:
            if sensor in data:
                sensor_info[sensor] = float(data[sensor])
        
        if sensor_info:
            result["key_sensors"] = sensor_info
        
        # 9. Add comparison between prediction methods (for analysis)
        if model_prediction is not None and has_pm25:
            result["comparison"] = {
                "model_prediction": int(model_prediction),
                "direct_prediction": int(direct_prediction),
                "final_prediction": int(final_prediction),
                "pm25_value": float(pm25_value),
                "confidence": float(probabilities[final_prediction])
            }
        
        return result, data
    except Exception as e:
        logging.error(f"❌ Error in prediction: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, data

# Load model and columns
model, model_columns = load_model()

# Connect to MongoDB Atlas
mongo_collection = connect_mongodb()

# Create Flask app
app = Flask(__name__)

@app.route('/savedata', methods=['POST'])
def save_data():
    try:
        # Receive JSON data from request
        data = request.json
        logging.info(f"📥 Received data from HTTP POST: {data}")
        
        # Check if data is valid
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Process data with ML model
        prediction_result, updated_data = predict_air_quality(data, model, model_columns)
        
        if prediction_result:
            # Save data to MongoDB
            try:
                if mongo_collection is None:
                    logging.error("❌ Cannot save data: No MongoDB connection")
                    return jsonify({"error": "MongoDB connection failed"}), 500
                    
                # Create document to save
                document = {
                    "timestamp": datetime.datetime.now(),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "sensor_data": updated_data,
                    "prediction": prediction_result
                }
                
                # Convert NumPy and other data types to Python native types
                document = json.loads(json.dumps(document, default=str))
                
                # Save to MongoDB
                result = mongo_collection.insert_one(document)
                
                logging.info(f"✅ Data saved to MongoDB Atlas successfully (ID: {result.inserted_id})")
                
                # Display prediction data
                aqi_class = prediction_result['aqi_class']
                aqi_label = prediction_result['aqi_label']
                method = prediction_result.get('method', 'unknown')
                
                logging.info(f"🔍 Air Quality Assessment Result (Method: {method}):")
                logging.info(f"   → Air Quality Level: {aqi_class} - {aqi_label}")
                
                # Send result back to Arduino
                return jsonify({
                    "success": True,
                    "message": "Data processed and saved successfully",
                    "prediction": prediction_result
                }), 200
                
            except Exception as e:
                logging.error(f"❌ Error saving data: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return jsonify({"error": "Failed to save data"}), 500
        else:
            return jsonify({"error": "Failed to process data"}), 500
            
    except Exception as e:
        logging.error(f"❌ Error processing request: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check if model and MongoDB connection are ready
    if model is None or model_columns is None:
        logging.error("Cannot proceed because model is missing. Please run train_model_rf.py first")
    elif mongo_collection is None:
        logging.warning("⚠️ Cannot connect to MongoDB, data will not be saved")
    
    # Show welcome message
    print("\n" + "="*60)
    print(" 🌟  Classroom Air Quality Assessment System  🌟")
    print("="*60)
    print("\n📡 Starting HTTP server to receive sensor data...")
    print(f"🔌 Listening on port 5001")
    print(f"🗄️  Connected to database: {MONGO_DB}")
    print("\n🔍 Waiting for data from ESP32...")
    print("="*60)
    print("📝 Usage instructions:")
    print("  1. Run data_cleaner.py to clean the data")
    print("  2. Run train_model_rf.py to train the model")
    print("  3. Run http_to_mongo.py (this file) to start the server")
    print("  4. Make sure ESP32 is connected to the same network")
    print("  5. Data will be saved to MongoDB and shown in server logs")
    print("="*60 + "\n")
    
    # Run Flask server
    logging.info("🚀 Starting HTTP server for air quality data")
    app.run(host='0.0.0.0', port=5001, debug=True)
