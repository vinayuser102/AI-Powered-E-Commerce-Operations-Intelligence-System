import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class SellerAnomalyEngine:
    def __init__(self, model_dir: str = "models", output_dir: str = "outputs", contamination: float = 0.05):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.contamination = contamination
        self.scaler_path = os.path.join(model_dir, "anomaly_scaler.pkl")
        self.model_path = os.path.join(model_dir, "isolation_forest.pkl")
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=self.contamination, random_state=42)

    def extract_seller_features(self, df_master: pd.DataFrame) -> pd.DataFrame:
        """Calculates delivery delays and aggregates seller-level statistics."""
        df = df_master.copy()
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

        delivered = df[df['order_status'] == 'delivered'].copy()
        delivered['delivery_delay_days'] = (
            delivered['order_delivered_customer_date'] - delivered['order_estimated_delivery_date']
        ).dt.days

        seller_stats = delivered.groupby('seller_id').agg(
            avg_delay=('delivery_delay_days', 'mean'),
            total_orders=('order_id', 'nunique'),
            avg_price=('price', 'mean'),
            total_revenue=('payment_value', 'sum')
        ).reset_index().dropna()

        return seller_stats

    def detect_anomalies(self, seller_stats: pd.DataFrame) -> pd.DataFrame:
        """Scales features, fits IsolationForest, and tags anomalies."""
        features = ['avg_delay', 'total_orders', 'avg_price', 'total_revenue']
        X = seller_stats[features]

        X_scaled = self.scaler.fit_transform(X)
        seller_stats['anomaly'] = self.model.fit_predict(X_scaled)
        seller_stats['anomaly_label'] = seller_stats['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        # Save model artifacts
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.model, self.model_path)
        print(f"✅ Scaler saved to {self.scaler_path}")
        print(f"✅ IsolationForest model saved to {self.model_path}")

        return seller_stats

    def save_anomaly_plot(self, seller_stats: pd.DataFrame, file_path: str = None):
        """Generates scatter plot highlighting seller anomalies."""
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'anomaly_detection.png')

        colors = seller_stats['anomaly'].map({1: 'steelblue', -1: 'red'})

        plt.figure(figsize=(10, 6))
        plt.scatter(
            seller_stats['avg_delay'], 
            seller_stats['total_revenue'],
            c=colors, alpha=0.6, edgecolors='white', linewidth=0.5
        )
        plt.title('SmartOps — Seller Anomaly Detection')
        plt.xlabel('Average Delivery Delay (days)')
        plt.ylabel('Total Revenue (BRL)')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        print(f"✅ Anomaly plot saved to {file_path}")