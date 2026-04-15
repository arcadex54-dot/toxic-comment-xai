import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print("Loading model...")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Loading dataset...")

data = pd.read_csv("dataset.csv")


print("Preparing data...")

X = vectorizer.transform(data["Text"])
y = data["IsToxic"]


print("Predicting...")

predictions = model.predict(X)


print("\nClassification Report:\n")

print(classification_report(y, predictions))


print("Generating confusion matrix...")

cm = confusion_matrix(y, predictions)


sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()