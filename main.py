from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")
feature_names = joblib.load("features.joblib")
cluster_labels = joblib.load("labels.joblib")

app = FastAPI(title="Store Clustering API")


class StoreFeatures(BaseModel):
    Marketing_Spend: float
    Store_Size: float
    Competitor_Price_Index: float


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Store Clustering API. Use the /cluster endpoint to get cluster predictions."
    }


@app.post("/cluster")
def assign_cluster(data: StoreFeatures):
    x = np.array([[getattr(data, feature) for feature in feature_names]])
    x_scaled = scaler.transform(x)
    cluster_id = int(kmeans.predict(x_scaled)[0])
    store_category = cluster_labels.get(cluster_id, "Unknown")
    return {
        "cluster_id": cluster_id,
        "store_category": store_category
    }
