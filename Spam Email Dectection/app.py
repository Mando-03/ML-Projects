from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('spam_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    emails = data.get('emails')

    if not emails:
        return jsonify({'error': 'No emails provided'}), 400

    email_counts = model.named_steps['vectorizer'].transform(emails)
    predictions = model.named_steps['nb'].predict(email_counts)

    results = ['spam' if pred else 'not spam' for pred in predictions]

    return jsonify({'predictions': results})


if __name__ == '__main__':
    app.run(debug=True)
