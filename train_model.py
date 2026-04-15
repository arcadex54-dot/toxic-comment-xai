import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


print("Loading dataset...")

data = pd.read_csv("dataset.csv")


print("Cleaning dataset...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    return text


# Use correct column name: Text
data["Text"] = data["Text"].apply(clean_text)


print("Vectorizing text...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["Text"])


# Use correct column name: IsToxic
y = data["IsToxic"]


print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training model...")

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)


print("Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


print("Saving model...")

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


print("Training complete ✅")