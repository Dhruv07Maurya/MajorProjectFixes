# This is the main brain of the autonomous greenhouse.
# It runs in a continuous loop to monitor and control the environment.
# It is designed to be run as a background service.

import time
import json
import sqlite3
from datetime import datetime
import RPi.GPIO as GPIO
import board
import busio
import adafruit_bme680
import adafruit_bh1750
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# --- CONFIGURATION ---
STATE_FILE = "state.json"
DB_FILE = "greenhouse.db"
PLANT_PROFILES_FILE = "plant_profiles.json"
MAIN_LOOP_INTERVAL = 2
DB_LOG_INTERVAL = 60
MOVING_AVERAGE_WINDOW = 5

# Hardcoded calibration values for simplicity.
SOIL_DRY_VALUE = 17500
SOIL_WET_VALUE = 9500

# --- HARDWARE PIN SETUP (BCM Mode) ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Relay Pins
INTAKE_FANS_PIN = 17
EXHAUST_FANS_PIN = 27
WATER_PUMP_PIN = 22
LED_LIGHT_PIN = 10

# Servo Pins
SERVO_1_PIN = 18
SERVO_2_PIN = 23

# Setup Pins
relay_pins = [INTAKE_FANS_PIN, EXHAUST_FANS_PIN, WATER_PUMP_PIN, LED_LIGHT_PIN]
for pin in relay_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)

GPIO.setup(SERVO_1_PIN, GPIO.OUT)
GPIO.setup(SERVO_2_PIN, GPIO.OUT)
servo1 = GPIO.PWM(SERVO_1_PIN, 50)
servo2 = GPIO.PWM(SERVO_2_PIN, 50)
servo1.start(0)
servo2.start(0)

# --- SENSOR INITIALIZATION ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x77)
    bh1750 = adafruit_bh1750.BH1750(i2c)
    ads = ADS.ADS1115(i2c)
    soil_sensor = AnalogIn(ads, 0)
    print("Controller: All sensors initialized successfully.")
except Exception as e:
    print(f"FATAL: Controller could not initialize sensors. Check wiring. Error: {e}")
    exit()

# --- HELPER FUNCTIONS ---

def load_json_file(filename, default_data={}):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_data

def save_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def log_to_db(table, data_dict):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        if table == 'sensor_readings':
            cursor.execute('''
                INSERT INTO sensor_readings (timestamp, temperature, humidity, light_intensity, soil_moisture, air_quality)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, data_dict['temperature'], data_dict['humidity'], data_dict['light'], data_dict['soil_moisture'], data_dict['air_quality']))
        elif table == 'action_logs':
            cursor.execute('INSERT INTO action_logs (timestamp, source, action) VALUES (?, ?, ?)', 
                           (timestamp, data_dict['source'], data_dict['action']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Controller: Database log error: {e}")

def control_relay(pin, state):
    gpio_state = GPIO.LOW if state else GPIO.HIGH
    if GPIO.input(pin) != gpio_state:
        GPIO.output(pin, gpio_state)

# --- MAIN LOGIC ---

def main():
    print("Controller: Starting main control loop...")
    
    state = load_json_file(STATE_FILE, {})
    soil_readings = []
    
    last_db_log_time = 0
    watering_start_time = 0
    ventilation_start_time = 0
    
    current_lid_angle = -1 # Use -1 to force an update on the first loop

    log_to_db('action_logs', {'source': 'SYSTEM', 'action': 'Controller script started.'})

    while True:
        try:
            current_time = time.time()
            now = datetime.now()

            temp_c = bme680.temperature
            humidity = bme680.humidity
            gas_resistance = bme680.gas
            light_lux = bh1750.lux
            
            soil_readings.append(soil_sensor.value)
            if len(soil_readings) > MOVING_AVERAGE_WINDOW:
                soil_readings.pop(0)
            smoothed_soil_raw = sum(soil_readings) / len(soil_readings) if soil_readings else 0

            state = load_json_file(STATE_FILE, {})
            profiles = load_json_file(PLANT_PROFILES_FILE, {})
            active_profile_name = state.get('active_profile_name')
            active_profile = profiles.get(active_profile_name)

            iaq_score = 0
            if (SOIL_DRY_VALUE - SOIL_WET_VALUE) != 0:
                soil_percent = max(0, min(100, ((SOIL_DRY_VALUE - smoothed_soil_raw) * 100) / (SOIL_DRY_VALUE - SOIL_WET_VALUE)))
            else:
                soil_percent = 0
            
            if active_profile:
                gas_baseline = active_profile.get('gas_baseline', 30000)
                if gas_baseline > 0:
                    iaq_score = min(100, (gas_resistance / gas_baseline) * 100)

            state['live_data'] = {
                'temperature': round(temp_c, 2), 'humidity': round(humidity, 1),
                'light': round(light_lux), 'soil_moisture': round(soil_percent, 1),
                'air_quality': round(iaq_score, 1), 'timestamp': datetime.utcnow().isoformat() + "Z"
            }

            if current_time - last_db_log_time >= DB_LOG_INTERVAL:
                log_to_db('sensor_readings', state['live_data'])
                last_db_log_time = current_time

            target_states = {}
            if state.get('system_mode') == 'AUTO' and active_profile:
                # --- AUTO LOGIC ---
                if now.day != state.get('last_light_check_day', now.day):
                    state['accumulated_light_minutes_today'] = 0
                    state['last_light_check_day'] = now.day
                    log_to_db('action_logs', {'source': 'AUTO', 'action': 'Daily light accumulator reset.'})

                temp_max = active_profile.get('temp_max_c', 30)
                hum_max = active_profile.get('humidity_max_percent', 75)
                is_too_hot_or_humid = temp_c > temp_max or humidity > hum_max
                
                target_states['intake_fans'] = False
                target_states['exhaust_fans'] = False
                target_states['lid_angle'] = 0

                if is_too_hot_or_humid:
                    target_states.update({'intake_fans': True, 'exhaust_fans': True, 'lid_angle': active_profile.get('lid_open_angle', 90)})
                elif temp_c < (temp_max - active_profile.get('hysteresis_temp', 2.0)) and humidity < (hum_max - active_profile.get('hysteresis_humidity', 5.0)):
                    if not state.get('is_ventilating', False):
                        target_states.update({'intake_fans': False, 'exhaust_fans': False, 'lid_angle': 0})
                
                # ... (the rest of your auto logic for water, light, etc. would go here)
            
            elif state.get('system_mode') == 'MANUAL':
                # --- MANUAL LOGIC ---
                target_states = {
                    'intake_fans': state.get('manual_intake_fans', False), 
                    'exhaust_fans': state.get('manual_exhaust_fans', False),
                    'water_pump': state.get('manual_water_pump', False), 
                    'led_light': state.get('manual_led_light', False),
                    'lid_angle': (active_profile.get('lid_open_angle', 90) if active_profile else 90) if state.get('manual_lid_open', False) else 0
                }
            
            else: # Auto mode but no profile, or any other case
                target_states = {'intake_fans': False, 'exhaust_fans': False, 'water_pump': False, 'led_light': False, 'lid_angle': 0}

            # --- APPLY ALL ACTUATOR STATES ---
            control_relay(INTAKE_FANS_PIN, target_states.get('intake_fans', False))
            control_relay(EXHAUST_FANS_PIN, target_states.get('exhaust_fans', False))
            control_relay(WATER_PUMP_PIN, target_states.get('water_pump', False))
            control_relay(LED_LIGHT_PIN, target_states.get('led_light', False))
            
            new_lid_angle = target_states.get('lid_angle', 0)
            if new_lid_angle != current_lid_angle:
                print(f"DEBUG: Lid angle command received. Moving from {current_lid_angle}° to {new_lid_angle}°")
                duty_cycle = 2 + (new_lid_angle / 18)
                servo1.ChangeDutyCycle(duty_cycle)
                servo2.ChangeDutyCycle(duty_cycle)
                time.sleep(1)
                servo1.ChangeDutyCycle(0)
                servo2.ChangeDutyCycle(0)
                current_lid_angle = new_lid_angle

            # --- SAVE FINAL STATE TO FILE ---
            # *** DEFINITIVE FIX: Add 'lid_open' boolean for the UI ***
            final_actuator_states = {
                'intake_fans': target_states.get('intake_fans', False),
                'exhaust_fans': target_states.get('exhaust_fans', False),
                'water_pump': target_states.get('water_pump', False),
                'led_light': target_states.get('led_light', False),
                'lid_angle': current_lid_angle,
                'lid_open': current_lid_angle > 0
            }
            state['live_actuator_states'] = final_actuator_states
            save_json_file(STATE_FILE, state)
            
            time.sleep(MAIN_LOOP_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Controller: An error occurred in the main loop: {e}")
            log_to_db('action_logs', {'source': 'ERROR', 'action': str(e)})
            time.sleep(10)
    
    GPIO.cleanup()
    print("Controller: GPIO cleanup complete. Exiting.")

if __name__ == '__main__':
    main()

