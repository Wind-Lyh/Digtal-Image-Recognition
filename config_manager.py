import json
import os

CONFIG_FILE = "config.json"

GLOBAL_CONFIG = {
    "blend_mode": "linear",
    "orb_features": 3000,
    "ransac_thresh": 5.0,
    "jpeg_quality": 95
}

def load_config():
    global GLOBAL_CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                GLOBAL_CONFIG.update(saved)
        except Exception as e:
            print(f"Failed to load config: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(GLOBAL_CONFIG, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")

def get_config(key, default_val=None):
    return GLOBAL_CONFIG.get(key, default_val)

def set_config(key, val):
    GLOBAL_CONFIG[key] = val

# 启动时自动加载
load_config()
