def explain_values(hr, bp_avg, spo2, temp):
    reasons = []
    
    if spo2 < 94:
        reasons.append("Low Oxygen Saturation (Hypoxia)")
    if hr > 100:
        reasons.append("Tachycardia (High Heart Rate)")
    elif hr < 60:
        reasons.append("Bradycardia (Low Heart Rate)")
    
    if bp_avg > 110:
        reasons.append("High Blood Pressure detected")
    
    if temp > 37.5:
        reasons.append("Pyrexia (Fever) detected")
    elif temp < 35.5:
        reasons.append("Hypothermia detected")

    if not reasons:
        reasons.append("Vitals are within standard ranges")
        
    return reasons
