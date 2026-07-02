import json
import os
import sys


def get_app_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


CONFIG_FILE = os.path.join(get_app_root(), "config.json")

GLOBAL_CONFIG = {
    "blend_mode": "linear",
    "orb_features": 3000,
    "ransac_thresh": 5.0,
    "jpeg_quality": 95
}


def load_config():
    global GLOBAL_CONFIG
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                GLOBAL_CONFIG.update(saved)
        else:
            save_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        save_config()


def save_config():
    try:
        parent_dir = os.path.dirname(CONFIG_FILE)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(GLOBAL_CONFIG, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")


def get_config(key, default_val=None):
    return GLOBAL_CONFIG.get(key, default_val)


def set_config(key, val):
    GLOBAL_CONFIG[key] = val


load_config()
