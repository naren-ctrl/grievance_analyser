import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess
import joblib

df = pd.read_csv("grievances.csv")
df["clean_text"] = df["text"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["department"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "dept_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved.")
