import joblib
from flask import Flask, request, jsonify,render_template
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH

app = Flask(__name__)
model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        
            lead_time=int(request.form['lead_time'])
            num_of_special_requests=int(request.form['num_of_special_requests'])
            average_price_per_room=float(request.form['average_price_per_room'])
            arrival_month=int(request.form['arrival_month'])
            arrival_date=int(request.form['arrival_date'])
            market_segment_type=int(request.form['market_segment_type'])
            no_of_weeks_nights=int(request.form['no_of_weeks_nights'])
            no_of_weekend_nights=int(request.form['no_of_weekend_nights'])
            type_of_meal_plan=int(request.form['type_of_meal_plan'])
            room_type_reserved=int(request.form['room_type_reserved'])


            features=np.array([[lead_time,num_of_special_requests,average_price_per_room,arrival_month,arrival_date,market_segment_type,no_of_weeks_nights,no_of_weekend_nights,type_of_meal_plan,room_type_reserved]])

            prediction=model.predict(features)

            return render_template('index.html', prediction_text='The predicted booking status is: {}'.format(prediction[0]))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)