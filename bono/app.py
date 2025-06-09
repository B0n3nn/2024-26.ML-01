from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():

    my_model = joblib.load('best_species_classifier.joblib')
    columns = ["fixed_acidity","residual_sugar","alcohol","density","quality_label"]
    data = request.get_json()
    df = pd.DataFrame(data['data'], columns=columns, index=[0])
    x = df.drop(columns=['quality_label'])
    start_time = time.perf_counter()
    y_pred = my_model.predict(x)
    end_time = time.perf_counter()

    inference_time = end_time - start_time

    return jsonify({
        "predictions": y_pred[0],
        "inference_time": inference_time
    })

if __name__ == '__main__':
    app.run(debug=True)