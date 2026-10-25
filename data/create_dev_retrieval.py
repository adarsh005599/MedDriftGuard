"""
Generates data/dev_retrieval.jsonl with 100 diagnostic questions
mapped to gold chunk IDs across all 7 medical topics.
Covers English, Devanagari Hindi, and Roman Hinglish.
"""

import json
from pathlib import Path

QUESTIONS = [
    # --- FEVER (med_fever_0001 .. 0004) ---
    ("What body temperature is considered a high fever in an adult?", "med_fever_0001", "fever", "en", "en"),
    ("Why is fever sometimes beneficial in fighting off infections?", "med_fever_0001", "fever", "en", "en"),
    ("Common signs and symptoms that accompany a fever", "med_fever_0002", "fever", "en", "en"),
    ("Why do children between 6 months and 5 years get febrile seizures?", "med_fever_0002", "fever", "en", "en"),
    ("Best home care and hydration drinks when having fever", "med_fever_0003", "fever", "en", "en"),
    ("Should I bundle up in blankets when having fever chills?", "med_fever_0003", "fever", "en", "en"),
    ("When should an adult consult a doctor for a fever lasting multiple days?", "med_fever_0004", "fever", "en", "en"),
    ("Emergency fever red flags like stiff neck and breathing difficulty", "med_fever_0004", "fever", "en", "en"),
    ("bukhar mein kitna temperature normal se jyada hota hai?", "med_fever_0001", "fever", "hinglish", "roman_hindi"),
    ("bukhar ke sath sir dard aur shivering hone ka kya matlab hai?", "med_fever_0002", "fever", "hinglish", "roman_hindi"),
    ("bukhar mein dehydration se bachne ke liye kya peena chahiye?", "med_fever_0003", "fever", "hinglish", "roman_hindi"),
    ("kya bukhar mein kambal odhna chahiye ya nahi?", "med_fever_0003", "fever", "hinglish", "roman_hindi"),
    ("agar bukhar 3 din se jyada rahe toh doctor ko kab dikhaye?", "med_fever_0004", "fever", "hinglish", "roman_hindi"),
    ("bukhar ke sath gardan akad jaye toh kya karein?", "med_fever_0004", "fever", "hinglish", "roman_hindi"),
    ("बुखार क्या होता है और यह क्यों आता है?", "med_fever_0001", "fever", "hi", "devanagari"),
    ("बुखार के साथ कंपकंपी और सिरदर्द के लक्षण", "med_fever_0002", "fever", "hi", "devanagari"),
    ("बुखार में घर पर पानी और ओआरएस की देखभाल", "med_fever_0003", "fever", "hi", "devanagari"),
    ("बुखार में कब तुरंत डॉक्टर या अस्पताल जाना जरूरी है?", "med_fever_0004", "fever", "hi", "devanagari"),

    # --- COUGH (med_cough_0001 .. 0003) ---
    ("Difference between a dry cough and a productive cough with phlegm", "med_cough_0001", "cough", "en", "en"),
    ("Common viral causes of acute bronchitis and coughing", "med_cough_0001", "cough", "en", "en"),
    ("Why chronic cough can be caused by GERD acid reflux and asthma", "med_cough_0001", "cough", "en", "en"),
    ("How warm fluids and honey help soothe an irritated throat cough", "med_cough_0002", "cough", "en", "en"),
    ("Why honey should never be given to infants under one year old", "med_cough_0002", "cough", "en", "en"),
    ("Using steam inhalation and humidifiers to relieve nighttime coughing", "med_cough_0002", "cough", "en", "en"),
    ("Red-flag cough symptoms like coughing up pink frothy phlegm or blood", "med_cough_0003", "cough", "en", "en"),
    ("When an unexplained cough lasts longer than three weeks", "med_cough_0003", "cough", "en", "en"),
    ("sukhi khansi aur balgam wali khansi mein kya antar hai?", "med_cough_0001", "cough", "hinglish", "roman_hindi"),
    ("gale ki khansi mein shahad aur garam paani ke fayde", "med_cough_0002", "cough", "hinglish", "roman_hindi"),
    ("chhote bachho ko shahad kyu nahi dena chahiye?", "med_cough_0002", "cough", "hinglish", "roman_hindi"),
    ("khansi ke sath khoon ya saans lene mein takleef ho toh kya karein?", "med_cough_0003", "cough", "hinglish", "roman_hindi"),
    ("sukhi khansi raat ko jyada kyu badh jati hai?", "med_cough_0002", "cough", "hinglish", "roman_hindi"),
    ("खांसी और गले में बलगम के घरेलू उपाय", "med_cough_0002", "cough", "hi", "devanagari"),
    ("खांसी में खून आना और सांस फूलना कब गंभीर है?", "med_cough_0003", "cough", "hi", "devanagari"),
    ("लंबे समय तक रहने वाली सूखी खांसी के कारण", "med_cough_0001", "cough", "hi", "devanagari"),

    # --- HEADACHE (med_headache_0001 .. 0003) ---
    ("What are the characteristics of a tension-type headache?", "med_headache_0001", "headache", "en", "en"),
    ("Migraine symptoms with throbbing pain, light sensitivity, and nausea", "med_headache_0001", "headache", "en", "en"),
    ("Home management of tension headaches through relaxation and dark room", "med_headache_0002", "headache", "en", "en"),
    ("Using cold pack on forehead or warm compress on neck for headache", "med_headache_0002", "headache", "en", "en"),
    ("What is a thunderclap headache and why is it an emergency?", "med_headache_0003", "headache", "en", "en"),
    ("Headache accompanied by stiff neck, fever, confusion, and seizure", "med_headache_0003", "headache", "en", "en"),
    ("sar ke aage bandha hua tight dard tension headache hota hai kya?", "med_headache_0001", "headache", "hinglish", "roman_hindi"),
    ("aadhe sar mein tez dard aur roshni se pareshaani migraine hai?", "med_headache_0001", "headache", "hinglish", "roman_hindi"),
    ("sar dard mein thandi patti ya andhere kamre mein aaram ke fayde", "med_headache_0002", "headache", "hinglish", "roman_hindi"),
    ("achanak tez dhamake jaisa sar dard ho toh kya karein?", "med_headache_0003", "headache", "hinglish", "roman_hindi"),
    ("माइग्रेन और सिरदर्द के प्रमुख लक्षण क्या हैं?", "med_headache_0001", "headache", "hi", "devanagari"),
    ("सिरदर्द से राहत पाने के घरेलू और प्राकृतिक उपाय", "med_headache_0002", "headache", "hi", "devanagari"),
    ("अचानक तेज सिरदर्द और गर्दन में अकड़न कब इमरजेंसी है?", "med_headache_0003", "headache", "hi", "devanagari"),

    # --- DENGUE AWARENESS (who_dengue_awareness_0001 .. 0004) ---
    ("How is dengue virus transmitted by infected Aedes mosquitoes?", "who_dengue_awareness_0001", "dengue awareness", "en", "en"),
    ("Symptoms of mild dengue fever including eye pain and breakbone aches", "who_dengue_awareness_0002", "dengue awareness", "en", "en"),
    ("Warning signs of severe dengue that emerge when fever drops", "who_dengue_awareness_0002", "dengue awareness", "en", "en"),
    ("Why oral rehydration and bed rest are essential for dengue care", "who_dengue_awareness_0003", "dengue awareness", "en", "en"),
    ("Preventing mosquito bites using mosquito nets and window screens", "who_dengue_awareness_0003", "dengue awareness", "en", "en"),
    ("Emergency hospital care for mucosal bleeding and severe abdominal pain", "who_dengue_awareness_0004", "dengue awareness", "en", "en"),
    ("dengue machhar ke katne se kaise phailta hai?", "who_dengue_awareness_0001", "dengue awareness", "hinglish", "roman_hindi"),
    ("dengue bukhar mein aankhon ke peeche dard aur body pain hota hai?", "who_dengue_awareness_0002", "dengue awareness", "hinglish", "roman_hindi"),
    ("dengue mein pet dard aur khoon aane ke warning signs", "who_dengue_awareness_0002", "dengue awareness", "hinglish", "roman_hindi"),
    ("dengue patient ko ghar par ORS aur paani kitna peena chahiye?", "who_dengue_awareness_0003", "dengue awareness", "hinglish", "roman_hindi"),
    ("dengue mein bleeding aur chakkar aane par hospital kab jayein?", "who_dengue_awareness_0004", "dengue awareness", "hinglish", "roman_hindi"),
    ("डेंगू बुखार के लक्षण और तेज बुखार की पहचान", "who_dengue_awareness_0002", "dengue awareness", "hi", "devanagari"),
    ("डेंगू में प्लेटलेट्स और गंभीर खतरे के संकेत", "who_dengue_awareness_0002", "dengue awareness", "hi", "devanagari"),
    ("डेंगू से बचाव और मच्छरों से सुरक्षा के उपाय", "who_dengue_awareness_0003", "dengue awareness", "hi", "devanagari"),
    ("गंभीर डेंगू में अस्पताल में तुरंत भर्ती कब कराएं?", "who_dengue_awareness_0004", "dengue awareness", "hi", "devanagari"),

    # --- HYPERTENSION AWARENESS (who_hypertension_awareness_0001 .. 0004) ---
    ("What blood pressure reading defines hypertension in adults?", "who_hypertension_awareness_0001", "hypertension awareness", "en", "en"),
    ("Why high blood pressure is referred to as a silent killer", "who_hypertension_awareness_0001", "hypertension awareness", "en", "en"),
    ("Symptoms of very high blood pressure like morning headaches and nosebleeds", "who_hypertension_awareness_0002", "hypertension awareness", "en", "en"),
    ("Long term complications of untreated hypertension on heart and kidneys", "who_hypertension_awareness_0002", "hypertension awareness", "en", "en"),
    ("Reducing dietary salt and sodium intake to lower blood pressure", "who_hypertension_awareness_0003", "hypertension awareness", "en", "en"),
    ("Recommended daily brisk walking and exercise for hypertension", "who_hypertension_awareness_0003", "hypertension awareness", "en", "en"),
    ("What defines a hypertensive crisis and requires calling 112 or 108?", "who_hypertension_awareness_0004", "hypertension awareness", "en", "en"),
    ("high blood pressure kitna reading aane par consider hota hai?", "who_hypertension_awareness_0001", "hypertension awareness", "hinglish", "roman_hindi"),
    ("high BP ko silent killer kyu kaha jata hai?", "who_hypertension_awareness_0001", "hypertension awareness", "hinglish", "roman_hindi"),
    ("blood pressure badhne par subah sar dard aur naak se khoon aana", "who_hypertension_awareness_0002", "hypertension awareness", "hinglish", "roman_hindi"),
    ("high BP control karne ke liye namak aur exercise ke niyam", "who_hypertension_awareness_0003", "hypertension awareness", "hinglish", "roman_hindi"),
    ("agar BP 180 ke upar chala jaye toh emergency mein kya karein?", "who_hypertension_awareness_0004", "hypertension awareness", "hinglish", "roman_hindi"),
    ("उच्च रक्तचाप (हाइपरटेंशन) के सामान्य लक्षण और कारण", "who_hypertension_awareness_0001", "hypertension awareness", "hi", "devanagari"),
    ("हाई बीपी को कम करने के लिए नमक और जीवनशैली में बदलाव", "who_hypertension_awareness_0003", "hypertension awareness", "hi", "devanagari"),
    ("ब्लड प्रेशर अचानक 180/120 होने पर आपातकालीन कदम", "who_hypertension_awareness_0004", "hypertension awareness", "hi", "devanagari"),

    # --- DIARRHOEA & DEHYDRATION (moh_diarrhoea_and_dehydration_0001 .. 0004) ---
    ("Clinical definition of diarrhoea as passage of loose or liquid stools", "moh_diarrhoea_and_dehydration_0001", "diarrhoea and dehydration", "en", "en"),
    ("Why loss of electrolytes is the main threat during acute diarrhoea", "moh_diarrhoea_and_dehydration_0001", "diarrhoea and dehydration", "en", "en"),
    ("Recognizing signs of mild to moderate dehydration like dry mouth", "moh_diarrhoea_and_dehydration_0002", "diarrhoea and dehydration", "en", "en"),
    ("Severe dehydration signs like sunken eyes, lethargy, and low blood pressure", "moh_diarrhoea_and_dehydration_0002", "diarrhoea and dehydration", "en", "en"),
    ("How to prepare homemade sugar salt solution when ORS packets are unavailable", "moh_diarrhoea_and_dehydration_0003", "diarrhoea and dehydration", "en", "en"),
    ("Easily digestible foods like khichdi and curd during diarrhoea recovery", "moh_diarrhoea_and_dehydration_0003", "diarrhoea and dehydration", "en", "en"),
    ("When diarrhoea with blood or persistent vomiting requires hospital visit", "moh_diarrhoea_and_dehydration_0004", "diarrhoea and dehydration", "en", "en"),
    ("dast hone par sharir mein dehydration kaise pehchane?", "moh_diarrhoea_and_dehydration_0002", "diarrhoea and dehydration", "hinglish", "roman_hindi"),
    ("ghar par ORS ka ghol ya cheeni namak paani kaise banayein?", "moh_diarrhoea_and_dehydration_0003", "diarrhoea and dehydration", "hinglish", "roman_hindi"),
    ("loose motion mein khichdi aur dahi khana chahiye kya?", "moh_diarrhoea_and_dehydration_0003", "diarrhoea and dehydration", "hinglish", "roman_hindi"),
    ("dast ke sath khoon ya lagatar ulti ho toh doctor ko kab dikhaye?", "moh_diarrhoea_and_dehydration_0004", "diarrhoea and dehydration", "hinglish", "roman_hindi"),
    ("दस्त और डिहाइड्रेशन के मुख्य लक्षण क्या हैं?", "moh_diarrhoea_and_dehydration_0001", "diarrhoea and dehydration", "hi", "devanagari"),
    ("घर पर ओआरएस घोल और तरल पदार्थ तैयार करने की विधि", "moh_diarrhoea_and_dehydration_0003", "diarrhoea and dehydration", "hi", "devanagari"),
    ("मल में खून और गंभीर निर्जलीकरण में तुरंत अस्पताल कब जाएं?", "moh_diarrhoea_and_dehydration_0004", "diarrhoea and dehydration", "hi", "devanagari"),

    # --- ACIDITY & GERD (moh_acidity_and_gerd_0001 .. 0004) ---
    ("What causes acid reflux when stomach acid flows into the esophagus?", "moh_acidity_and_gerd_0001", "acidity and gerd", "en", "en"),
    ("Complications of chronic GERD occurring more than twice a week", "moh_acidity_and_gerd_0001", "acidity and gerd", "en", "en"),
    ("Heartburn symptoms after meals and sour liquid regurgitation", "moh_acidity_and_gerd_0002", "acidity and gerd", "en", "en"),
    ("Dietary adjustments to avoid acidic trigger foods and oily gravies", "moh_acidity_and_gerd_0003", "acidity and gerd", "en", "en"),
    ("Why elevating the head of bed by 6 inches prevents nighttime acid reflux", "moh_acidity_and_gerd_0003", "acidity and gerd", "en", "en"),
    ("Distinguishing between harmless heartburn and serious cardiac chest pain", "moh_acidity_and_gerd_0004", "acidity and gerd", "en", "en"),
    ("pet mein gas aur seene mein jalan acidity ke lakshan", "moh_acidity_and_gerd_0002", "acidity and gerd", "hinglish", "roman_hindi"),
    ("khana khane ke baad seene mein khatti dakar aur jalan kyu hoti hai?", "moh_acidity_and_gerd_0002", "acidity and gerd", "hinglish", "roman_hindi"),
    ("acidity se bachne ke liye tel masale aur khana khane ke baad ke niyam", "moh_acidity_and_gerd_0003", "acidity and gerd", "hinglish", "roman_hindi"),
    ("seene ki jalan aur heart attack ke dard mein antar kaise karein?", "moh_acidity_and_gerd_0004", "acidity and gerd", "hinglish", "roman_hindi"),
    ("पेट में एसिडिटी और सीने में जलन के प्रमुख कारण", "moh_acidity_and_gerd_0001", "acidity and gerd", "hi", "devanagari"),
    ("एसिडिटी और गैस से बचने के लिए खानपान में सावधानियां", "moh_acidity_and_gerd_0003", "acidity and gerd", "hi", "devanagari"),
    ("हार्ट बर्न और दिल के दौरे (हार्ट अटैक) के दर्द में अंतर", "moh_acidity_and_gerd_0004", "acidity and gerd", "hi", "devanagari"),
]

def main():
    target_path = Path("data/dev_retrieval.jsonl")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with target_path.open("w", encoding="utf-8") as f:
        for idx, (query, gold_id, topic, lang, script) in enumerate(QUESTIONS, start=1):
            record = {
                "query_id": f"dev_q_{idx:03d}",
                "query": query,
                "gold_chunk_id": gold_id,
                "topic": topic,
                "language": lang,
                "script": script,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(QUESTIONS)} retrieval dev evaluation records in {target_path}")

if __name__ == "__main__":
    main()
