def check_drugs(drugs, diseases, allergies):
    warnings = []
    recommendations = []
    
    # Example logic - you can expand this with your specific medical rules
    drugs = [d.strip().lower() for d in drugs if d.strip()]
    diseases = [dis.strip().lower() for dis in diseases if dis.strip()]
    allergies = [a.strip().lower() for a in allergies if a.strip()]

    # Basic Safety Rules
    if "aspirin" in drugs and "ulcer" in diseases:
        warnings.append("⚠️ WARNING: Aspirin may aggravate gastric ulcers.")
    
    if "metformin" in drugs and "kidney" in diseases:
        warnings.append("⚠️ WARNING: Metformin requires careful monitoring in kidney conditions.")

    if any(a in drugs for a in allergies):
        warnings.append("🚨 CRITICAL: Patient is allergic to one of the entered medications!")

    if not warnings:
        warnings.append("✅ No immediate interactions detected in the local database.")
    
    recommendations.append("Ensure patient remains hydrated.")
    recommendations.append("Review full medication history with a pharmacist.")

    return warnings, recommendations
