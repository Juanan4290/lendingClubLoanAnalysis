from flask import Flask, jsonify, request
from src.utils import api_predict

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def apicall():
    """
    API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        # getting test set
        test_json = request.get_json()
        # predicting...
        scores = api_predict(test_json)

    except Exception as e:
        raise e

    if scores.empty:
        return(bad_request())
    else:
        print("Done!")
        responses = jsonify(predictions = scores.to_json(orient="records"))
        responses.status_code = 200

        return (responses)