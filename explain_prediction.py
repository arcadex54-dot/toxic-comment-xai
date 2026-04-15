import pickle
from lime.lime_text import LimeTextExplainer


print("Loading model...")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


explainer = LimeTextExplainer(class_names=["Non-toxic", "Toxic"])


def predict_proba(texts):

    vectors = vectorizer.transform(texts)

    return model.predict_proba(vectors)


comment = input("Enter a comment: ")


explanation = explainer.explain_instance(
    comment,
    predict_proba
)


print("\nImportant words influencing prediction:\n")

for word, weight in explanation.as_list():

    print(word, ":", weight)