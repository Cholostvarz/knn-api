
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Inicializácia aplikácie
app = Flask(__name__)

# Načítanie dát a trénovanie modelu pri štarte servera
df = pd.read_csv("ESCO_Kompetencie_Final.csv")
skill_columns = [f'k{i}' for i in range(1, 26)]

# Odstrániť stĺpce s nulovými hodnotami
non_zero_columns = df[skill_columns].loc[:, (df[skill_columns] != 0).any(axis=0)].columns
skill_columns = non_zero_columns

# Normalizácia dát
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[skill_columns] = scaler.fit_transform(df[skill_columns])

# Tréning KNN modelu
knn_model = NearestNeighbors(n_neighbors=5, metric='manhattan')
knn_model.fit(df_scaled[skill_columns].values)

@app.route('/')
def index():
    return "KNN odporúčací systém beží z CSV."

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json["responses"]
        if len(user_input) != 25:
            return jsonify({"error": "Očakáva sa presne 25 hodnôt."}), 400

        user_vector = np.array([float(val) for val in user_input])
        weighted_vector = np.where(user_vector == 0, 0, (user_vector / 5) ** 1.5)

        input_df = pd.DataFrame([weighted_vector], columns=[f'k{i}' for i in range(1, 26)])
        input_df = input_df[skill_columns]
        input_scaled = scaler.transform(input_df)

        distances, indices = knn_model.kneighbors(input_scaled)
        recommendations = df.iloc[indices[0]]["Zamestnanie"].tolist()

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
