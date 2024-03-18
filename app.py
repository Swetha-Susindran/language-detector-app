from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Load the model and necessary data
data = pd.read_csv("language_detection.csv")
X = data["Text"]
y = data["Language"]
le = LabelEncoder()
y = le.fit_transform(y)
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    text = text.lower()
    data_list.append(text)
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = MultinomialNB()
model.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    return render_template('result.html', language=lang[0])

if __name__ == '__main__':
    app.run(debug=True)
