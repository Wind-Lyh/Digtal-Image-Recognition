import json
import os
import uuid

HISTORY_FILE = "history.json"
HISTORY_IMG_DIR = os.path.join("output", "history_images")

def init_env():
    if not os.path.exists(HISTORY_IMG_DIR):
        os.makedirs(HISTORY_IMG_DIR, exist_ok=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load history: {e}")
    return []

def save_history(history_list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, indent=4)
    except Exception as e:
        print(f"Failed to save history: {e}")

def get_statistics():
    history = load_history()
    total = len(history)
    success = sum(1 for item in history if item.get("status") == "success")
    success_rate = (success / total * 100) if total > 0 else 0.0
    
    total_time = sum(item.get("duration", 0) for item in history if item.get("status") == "success")
    avg_time = (total_time / success) if success > 0 else 0.0
    
    return {
        "total": total,
        "success_rate": round(success_rate, 1),
        "avg_time": round(avg_time, 2)
    }

init_env()
