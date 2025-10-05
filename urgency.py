def detect_urgency(text):
    urgent_keywords = ["urgent", "emergency", "immediately", "danger", "severe", "critical"]
    if any(word in text.lower() for word in urgent_keywords):
        return "Urgent"
    return "Normal"
