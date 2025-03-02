from flask import Flask, render_template, request, flash, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os
from pest_info import pest_info
from pesticide_recommendations import pesticide_recommendations

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "4f7c23a34b5e4c8db0ffac23d1e8a978"  # For flash messages

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names
class_names = [
    'Adristyrannus', 'Aleurocanthus spiniferus', 'Ampelophaga', 'Aphis citricola Vander Goot',
    'Apolygus lucorum', 'Bactrocera tsuneonis', 'Beet spot flies', 'Black hairy',
    'Brevipoalpus lewisi McGregor', 'Ceroplastes rubens', 'Chlumetia transversa',
    'Chrysomphalus aonidum', 'Cicadella viridis', 'Cicadellidae', 'Colomerus vitis',
    'Dacus dorsalis(Hendel)', 'Dasineura sp', 'Deporaus marginatus Pascoe',
    'Erythroneura apicalis', 'Field Cricket', 'Fruit piercing moth', 'Gall fly',
    'Icerya purchasi Maskell', 'Indigo caterpillar', 'Jute Stem Weevil', 'Jute aphid',
    'Jute hairy', 'Jute red mite', 'Jute semilooper', 'Jute stem girdler',
    'Jute stick insect', 'Lawana imitata Melichar', 'Leaf beetle', 'Limacodidae',
    'Locust', 'Locustoidea', 'Lycorma delicatula', 'Mango flat beak leafhopper',
    'Mealybug', 'Miridae', 'Nipaecoccus vastalor', 'Panonchus citri McGregor',
    'Papilio xuthus', 'Parlatoria zizyphus Lucus', 'Pest_Dataset',
    'Phyllocnistis citrella Stainton', 'Phyllocoptes oleiverus ashmead',
    'Pieris canidia', 'Pod borer', 'Polyphagotars onemus latus', 'Potosiabre vitarsis'
]

# Load the trained model
def load_model(model_path, num_classes):
    model = models.convnext_base(pretrained=False)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Predict the class with accuracy
def predict_image(model, image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence = confidence.item() * 100
    return predicted_class, confidence

# Load the model
model_path = "best_model.pth"  # Update with your model path
model = load_model(model_path, len(class_names))

# Route for welcome page
@app.route('/', methods=['GET', 'POST'])
def welcome():
    return render_template('welcome.html')

# Route to handle user input from welcome page
@app.route('/home', methods=['POST'])
def home():
    name = request.form.get('name', 'User')
    language = request.form.get('language', 'English')
    return render_template('index.html', name=name, language=language)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is provided
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("No file selected. Please upload an image.")
        return redirect('/home')

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Predict the class using the trained model
    predicted_class, confidence = predict_image(model, file_path)
    pest_name = predicted_class

    # Get pest details from pest_info
    pest_details = pest_info.get(pest_name, {})
    english_name = pest_details.get('english_name', 'Unknown Pest')
    tamil_name = pest_details.get('tamil_name', 'Unknown Pest')
    english_description = pest_details.get('english_description', 'No description available.')
    tamil_description = pest_details.get('tamil_description', 'No description available.')
    english_prevention = pest_details.get('english_prevention', 'No prevention available.')
    tamil_prevention = pest_details.get('tamil_prevention', 'No prevention available.')

    # Get pesticide recommendations
    pesticide_details = pesticide_recommendations.get(pest_name, {})
    pesticide_name = pesticide_details.get('pesticide_name', 'No pesticide recommendation available.')
    tamil_pesticide = pesticide_details.get('tamil_pesticide', 'No Tamil pesticide recommendation available.')
    tamil_pesticide_description = pesticide_details.get('tamil_description', 'No Tamil description available.')

    # Get language from form submission
    language = request.form.get('language', 'English')

    # Render results on the index page
    return render_template(
        'index.html',
        name=request.form.get('name', 'User'),
        language=language,
        pest_name=pest_name,
        english_name=english_name,
        tamil_name=tamil_name,
        english_description=english_description,
        tamil_description=tamil_description,
        english_prevention=english_prevention,
        tamil_prevention=tamil_prevention,
        pesticide_name=pesticide_name,
        tamil_pesticide=tamil_pesticide,
        tamil_pesticide_description=tamil_pesticide_description,
        confidence=f"{confidence:.2f}%"
    )

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)