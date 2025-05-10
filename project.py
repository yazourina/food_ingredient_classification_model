from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 51)
model.load_state_dict(torch.load('fruits_vegetables_51.pth', map_location=device))
model.to(device)
model.eval()

class_names = [
    'Amaranth', 'Apple', 'Banana', 'Beetroot', 'Bell pepper', 'Bitter Gourd',
    'Blueberry', 'Bottle Gourd', 'Broccoli', 'Cabbage', 'Cantaloupe', 'Capsicum',
    'Carrot', 'Cauliflower', 'Chilli pepper', 'Coconut', 'Corn', 'Cucumber',
    'Dragon_fruit', 'Eggplant', 'Fig', 'Garlic', 'Ginger', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Okra', 'Onion', 'Orange', 'Paprika', 'Pear', 'Peas',
    'Pineapple', 'Pomegranate', 'Potato', 'Pumpkin', 'Raddish', 'Raspberry',
    'Ridge Gourd', 'Soy beans', 'Spinach', 'Spiny Gourd', 'Sponge Gourd',
    'Strawberry', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip', 'Watermelon'
]

data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file.stream)
    img = data_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        prediction = class_names[preds[0]]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)