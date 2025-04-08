
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Načítanie dát
df = pd.read_csv("ESCO_Agregacia_Kompetencie.csv")
skill_columns = [f"k{i}" for i in range(1, 26)]

# Normalizácia vstupných dát
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[skill_columns] = scaler.fit_transform(df[skill_columns])
model = NearestNeighbors(n_neighbors=5, metric="manhattan")
model.fit(df_scaled[skill_columns])

@app.route("/")
def home():
    return "Odporúčací systém API"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    responses = data.get("responses")
    if not responses or len(responses) != 25:
        return jsonify({"error": "Invalid input"}), 400

    input_scaled = scaler.transform([responses])
    distances, indices = model.kneighbors(input_scaled)

    results = df.iloc[indices[0]][["ID_zam", "Zamestnanie"]].to_dict(orient="records")
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
