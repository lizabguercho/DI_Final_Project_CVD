def categorize_bp(row):
    systolic = row["ap_hi"]
    diastolic = row["ap_lo"]

    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "Hypertension Stage 1"
    elif (140 <= systolic < 180) or (90 <= diastolic < 120):
        return "Hypertension Stage 2"
    elif systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis"
    else:
        return "Uncategorized"
