markdown
Copy code
# Spam Email Classifier

This project demonstrates a simple spam email classifier using the Multinomial Naive Bayes algorithm. It includes scripts for training the model, saving it, and deploying it as both a Flask API and a Streamlit web application.

## Overview

The project consists of the following components:

1. **Training Script**: `train.py` - This script loads the dataset, preprocesses the text data, trains the Multinomial Naive Bayes classifier, and saves the trained model to a file.

2. **Flask API**: `app.py` - This script serves as a Flask API for making predictions with the trained model. It provides endpoints for both single predictions and batch predictions.

3. **Streamlit Web Application**: `streamlit_app.py` - This script implements a simple web interface using Streamlit. Users can input text and get predictions on whether the input emails are spam or not.

4. **Model File**: `spam_classifier.pkl` - This is the trained model file saved using joblib.

## Usage

### Training the Model

To train the model, run the `train.py` script:

```sh
python train.py
This script will preprocess the dataset, train the model, and save it as spam_classifier.pkl.

Running the Flask API
To run the Flask API, execute the following command:

sh
python app.py
This will start the Flask server, and you can make predictions by sending HTTP requests to the appropriate endpoints.

Running the Streamlit Web Application
To run the Streamlit web application, use the following command:
http://127.0.0.1:5000/predict


{
    "emails": [
        "Hey mohamd, can we get together to watch football game tomorrow?",
        "Upto 20% discount on parking, exclusive offer just for you. Don't miss this reward!"
    ]
}


Requirements
Ensure you have the following dependencies installed:

Python 3.x
Flask
Streamlit
Scikit-learn
Joblib
Pandas
You can install these dependencies using pip:

sh
Copy code
pip install flask streamlit scikit-learn joblib pandas