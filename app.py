from flask import Flask, request, render_template
import joblib
import pandas as pd
from preprocess import preprocess
from urgency import detect_urgency
from duplicate_check import is_duplicate

app = Flask(__name__)

model = joblib.load("dept_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("grievances.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['complaint']
    clean_text = preprocess(text)
    vec = vectorizer.transform([clean_text])
    dept = model.predict(vec)[0]
    urgency = detect_urgency(text)
    is_dup = is_duplicate(text, df['text'].tolist())

    return render_template('index.html', 
        result=True, complaint=text,
        department=dept, urgency=urgency,
        duplicate="Yes" if is_dup else "No")

if __name__ == '__main__':
    app.run(debug=True)
