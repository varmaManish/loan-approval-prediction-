from flask import Flask , render_template, request, jsonify
import pickle as pkl
import numpy as np
import pandas as pd

# Load the trained model
model_path = "model.pkl"
with open(model_path, 'rb') as f:
    model=pkl.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        Dependents = float(request.form.get('Dependents'))
        Self_Employed = float(request.form.get('Self_Employed'))
        ApplicantIncome = float(request.form.get('ApplicantIncome'))
        CoapplicantIncome = float(request.form.get('CoapplicantIncome'))
        LoanAmount = float(request.form.get('LoanAmount'))
        Loan_Amount_Term = float(request.form.get('Loan_Amount_Term'))
        Credit_History = float(request.form.get('Credit_History'))
        # Add more features as needed based on the model input
    except ValueError:
        return "Invalid input. Please enter numeric values."

    # Create an array for the model
    features = pd.DataFrame([[Dependents,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History]],
                            columns=['Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History'])  # Adjust based on the number of features
    prediction = model.predict(features)
    output = 'approved' if prediction[0] == 1 else 'REJECT'
    return render_template('index.html', prediction_text=' {}'.format(output))
if __name__ == '__main__':
    app.run(debug=True)
