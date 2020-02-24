import pickle

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__,)
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    new = pd.DataFrame({
        'player_height': [args.get('player_height')],
        'country': [args.get('country')],
        'draft_year': [args.get('draft_year')],
        'draft_round': [args.get('draft_round')],
        'draft_number': [args.get('draft_number')]})

    prediction = (pipe.predict(new)[0])
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
