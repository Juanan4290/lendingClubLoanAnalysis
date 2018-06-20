from flask import Flask, jsonify, request
from src.model_launcher import api_predict

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def apicall():
    """
    API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    print("Starting process...")
    try:
        # getting test set
        test_json = request.get_json()

        # predicting...
        scores = api_predict(test_json)

    except ValueError as ex:
        out = {'code': 400, 'status': 'Bad request. Json body expected.', 'response': {}}
    except Exception as ex:
        out = {'code': 500, 'status': 'Unexpected error: %s' % str(ex), 'response': {}}
    else:
        out = {'code': 200, 'status': 'OK', 'response': {'score': scores}}        
        print("Done! Thanks for using Loan Defaul Prediction System API ;)")
    
    return jsonify(out)