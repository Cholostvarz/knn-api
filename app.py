
import numpy as np
import pandas as pd
import pymysql
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Flask aplikácia
app = Flask(__name__)

# Pripojenie k MySQL databáze
def load_data_from_db():
    conn = pymysql.connect(
        host='sql6.webzdarma.cz',
        user='zamestnaniee2288',
        password='6a6S&*Auk%)14gj(pyC,',
        database='zamestnaniee2288',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT ek.*, ez.nazov_zamestnania
            FROM ESCO_Agregacia_Kompetencie ek
            JOIN ESCO_Zamestnanie ez ON ek.ID_zam = ez.pk
        """)
        result = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(result)
    return df

df = load_data_from_db()
skill_categories = [f'k{i}' for i in range(1, 26)]

# Odstránenie irelevantných kategórií
non_zero_columns = df[skill_categories].loc[:, (df[skill_categories] != 0).any(axis=0)].columns
skill_categories = non_zero_columns

# Normalizácia datasetu (0–1)
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[skill_categories] = scaler.fit_transform(df[skill_categories])

# KNN model
knn_model = NearestNeighbors(n_neighbors=5, metric='manhattan')
knn_model.fit(df_normalized[skill_categories].values)

@app.route('/')
def index():
    return "KNN odporúčací systém API je pripravený."

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json["responses"]
    user_vector = np.array([float(val) for val in user_input])
    
    # škála bola 0–5 → normalizácia + váženie
    weighted_vector = np.where(user_vector == 0, 0, (user_vector / 5) ** 1.5)

    # Prispôsobenie vektoru na rovnakú dimenziu
    vector_df = pd.DataFrame([weighted_vector], columns=[f'k{i}' for i in range(1, 26)])
    vector_df = vector_df[skill_categories]
    user_vector_scaled = scaler.transform(vector_df)

    distances, indices = knn_model.kneighbors(user_vector_scaled)
    top_jobs = df.iloc[indices[0]]['nazov_zamestnania'].tolist()
    return jsonify(top_jobs)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
