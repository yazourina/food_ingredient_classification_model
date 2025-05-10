# food_ingredient_classification_model
Fruit and Vegetable Classifier API
This project is a Flask-based web API for classifying fruits and vegetables using a deep learning model. The model is based on ResNet50 and trained to identify 51 different types of fruits and vegetables. The API accepts an image file and returns the predicted class label.

Features
Classify a wide range of fruits and vegetables (51 classes).

Predict the class of an image uploaded via an HTTP POST request.

Use of PyTorch and pre-trained ResNet50 model.

Installation
Prerequisites
Ensure you have the following installed:

Python 3.6+

PyTorch (CUDA-enabled if using a GPU)

Flask

Pillow

Torchvision

You can install these dependencies using pip:

bash
Copy code
pip install torch torchvision flask pillow
Clone the Repository
bash
Copy code
git clone <repository_url>
cd <repository_directory>
Setup the Model
Download the pre-trained model weights (fruits_vegetables_51.pth) and place it in the root directory of the project.

Usage
Run the Flask application:

bash
Copy code
python app.py
The server will start running locally at http://127.0.0.1:5000/.

To classify an image, send a POST request to the /predict endpoint with a file parameter:

bash
Copy code
curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict
Example Response
json
Copy code
{
  "prediction": "Apple"
}
API Endpoint
POST /predict
Accepts a file parameter (image) and returns a prediction in the form of a JSON object.

Request Format
POST request

Content-Type: multipart/form-data

Parameter: file (Image file to classify)

Response Format
The response is a JSON object with the following structure:

json
Copy code
{
  "prediction": "<predicted_class>"
}
Model Details
The model uses a custom-trained ResNet50 model with 51 output classes, corresponding to various fruits and vegetables. The model is pre-trained on a custom dataset of fruits and vegetables images.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
PyTorch

Flask

TorchVision

Pillow

Dataset is taken from https://www.kaggle.com/code/sunnyagarwal427444/food-ingredient-classifiication-model
