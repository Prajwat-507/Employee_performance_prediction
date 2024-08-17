from flask import Flask, render_template, request
import pandas as pd
import pickle

model = pickle.load(open('dataset.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    return render_template('predict.html')


@app.route('/evaluate', methods=['POST','GET'])
def evaluate():
    quarter = request.form['quarter']
    department = request.form['department']
    day = request.form['day']
    team = int(request.form['team'])
    over_time = int(request.form['over_time'])
    targeted_productivity = float(request.form['targeted_productivity'])
    incentive = int(request.form['incentive'])
    smv = float(request.form['smv'])
    month = request.form['month']
    idle_time = float(request.form['idle_time'])
    idle_men = int(request.form['idle_men'])
    no_of_style_change = int(request.form['no_of_style_change'])
    no_of_workers = float(request.form['no_of_workers'])

    
    new_data = {
    'quarter': [quarter],
    'department': [department],
    'day': [day],
    'team': [team],
    'over_time': [over_time],
    'targeted_productivity': [targeted_productivity],
    'incentive': [incentive],
    'smv': [smv],
    'month': [month],
    'idle_time': [idle_time],
    'idle_men': [idle_men],
    'no_of_style_change': [no_of_style_change],
    'no_of_workers': [no_of_workers]
    }

    # Convert the new data into a pandas DataFrame
    new_df = pd.DataFrame(new_data)

    ans = pipe.predict(new_df)


    if(ans <= 0.3):
        text = 'The employee has low productivity.'
    elif(ans > 0.3 and ans <= 0.7 ):
        text =  'The employee has average productivity.'
    elif(ans > 0.7 and ans <= 0.89):
        text = 'The employee is Highly productive.'
    else:
        text = 'The employee has best productivity of all.'

    
    return render_template('result.html', data=text)
    

if __name__ == '__main__':
    app.run(debug=True)