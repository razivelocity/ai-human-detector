from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text_input = request.form["text_input"]

        # Transform text
        X_text = vectorizer.transform([text_input])

        # Dummy numeric features (set to 0 for user input)
        numeric_features = np.zeros((1, 7))

        # Combine
        import scipy.sparse as sp
        X = sp.hstack([X_text, numeric_features])

        prediction = model.predict(X)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)