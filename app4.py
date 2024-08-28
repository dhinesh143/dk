import re
import spacy
from transformers import pipeline
from flask import Flask, request, jsonify, render_template

# Load spaCy's pre-trained model and Zero-shot classification model
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", clean_up_tokenization_spaces= "True" )

# Define categories
customer_requirements = {
    "Car Type": ["Hatchback", "SUV", "Sedan"],
    "Fuel Type": ["Petrol", "Diesel", "Electric"],
    "Color": ["White", "Black", "Red", "Blue", "Silver", "Gray", "Green"],
    "Distance Travelled": [],
    "Make Year": [],
    "Transmission Type": ["Automatic", "Manual"]
}

company_policies = [
    "Free RC Transfer",
    "5-Day Money Back Guarantee",
    "Free RSA for One Year",
    "Return Policy"
]

customer_objections = [
    "Refurbishment Quality",
    "Car Issues",
    "Price Issues",
    "Customer Experience Issues"
]

def preprocess_text(text):
    """Clean the text by removing unwanted characters."""
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'[^A-Za-z0-9,.!? ]+', '', text)  # Remove special characters
    return text

def extract_customer_requirements(transcript):
    result = {}
    doc = nlp(transcript)
    
    for category, options in customer_requirements.items():
        if options:
            prediction = classifier(transcript, options)
            result[category] = prediction['labels'][0] if prediction['scores'][0] > 0.5 else None
        else:
            if category == "Make Year":
                years = [ent.text for ent in doc.ents if re.match(r"\b(19|20)\d{2}\b", ent.text)]
                result[category] = years if years else None
            elif category == "Distance Travelled":
                distances = [ent.text for ent in doc.ents if re.match(r"\d+(,\d{3})*(\.\d+)?\s*(km|kilometers|miles)", ent.text)]
                result[category] = distances if distances else None
    
    return result

def extract_company_policies(transcript):
    result = []
    prediction = classifier(transcript, company_policies)
    for i, policy in enumerate(prediction['labels']):
        if prediction['scores'][i] > 0.7:
            result.append(policy)
    return result

def extract_customer_objections(transcript):
    result = []
    prediction = classifier(transcript, customer_objections)
    for i, objection in enumerate(prediction['labels']):
        if prediction['scores'][i] > 0.7:
            result.append(objection)
    return result

def process_transcript(transcript, conversation_id):
    customer_requirements = extract_customer_requirements(transcript)
    company_policies = extract_company_policies(transcript)
    customer_objections = extract_customer_objections(transcript)

    return {
        "conversation_id": conversation_id,
        "customer_requirements": customer_requirements,
        "company_policies_discussed": company_policies,
        "customer_objections": customer_objections
    }

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    transcript = file.read().decode('utf-8')
    
    # Process the transcript
    cleaned_transcript = preprocess_text(transcript)
    conversation_id = "transcript_001"  # You can replace this with dynamic IDs
    response = process_transcript(cleaned_transcript, conversation_id)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
