
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Načítanie datasetu
df = pd.read_csv("ESCO_Kompetencie_Final.csv")

# Automatické získanie stĺpcov zručností od 3. stĺpca (rovnako ako v appKNN2)
skill_columns = df.columns[3:]

# Odstránenie irelevantných (nulových) kategórií
non_zero_columns = df[skill_columns].loc[:, (df[skill_columns] != 0).any(axis=0)].columns
skill_columns = non_zero_columns

# Normalizácia datasetu
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[skill_columns] = scaler.fit_transform(df[skill_columns])

# Tréning modelu
knn_model = NearestNeighbors(n_neighbors=5, metric='manhattan')
knn_model.fit(df_scaled[skill_columns].values)

@app.route('/')
def index():
    return "Optimalizovaný KNN model je pripravený."

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json["responses"]
        if len(user_input) != len(df.columns) - 3:
            return jsonify({"error": f"Očakáva sa {len(df.columns) - 3} hodnôt."}), 400

        user_vector = np.array([float(val) for val in user_input])
        weighted_vector = user_vector ** 1.5

        # Normalizuj rovnako ako dataset
        input_df = pd.DataFrame([weighted_vector], columns=skill_columns)
        input_scaled = scaler.transform(input_df)

        distances, indices = knn_model.kneighbors(input_scaled)
        recommendations = df.iloc[indices[0]]["Zamestnanie"].tolist()

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
