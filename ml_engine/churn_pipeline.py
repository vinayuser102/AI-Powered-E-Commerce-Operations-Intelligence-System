import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CustomerBehaviorEngine:
    def __init__(self, model_dir: str = "models", output_dir: str = "outputs"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.scaler_path = os.path.join(model_dir, "rfm_scaler.pkl")
        self.model_path = os.path.join(model_dir, "churn_model.pkl")
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    def calculate_rfm(self, df_master: pd.DataFrame) -> pd.DataFrame:
        """Calculates RFM metrics, scores, and customer segments."""
        df_master = df_master.copy()
        df_master['order_purchase_timestamp'] = pd.to_datetime(df_master['order_purchase_timestamp'])
        
        reference_date = df_master['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
        delivered = df_master[df_master['order_status'] == 'delivered'].copy()

        rfm = delivered.groupby('customer_id').agg(
            Recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
            Frequency=('order_id', 'nunique'),
            Monetary=('payment_value', 'sum')
        ).reset_index()

        # Score each dimension 1-5
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

        rfm['RFM_Score'] = (rfm['R_Score'].astype(int) + 
                            rfm['F_Score'].astype(int) + 
                            rfm['M_Score'].astype(int))

        def assign_segment(score):
            if score >= 13: return 'Champions'
            elif score >= 10: return 'Loyal'
            elif score >= 7: return 'At Risk'
            elif score >= 4: return 'Needs Attention'
            else: return 'Lost'

        rfm['Segment'] = rfm['RFM_Score'].apply(assign_segment)
        rfm['Churned'] = (rfm['Recency'] > 180).astype(int)
        
        return rfm

    def save_segment_plot(self, rfm: pd.DataFrame, file_path: str = None):
        """Generates and saves the RFM segmentation pie chart."""
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'customer_segments.png')
            
        seg_counts = rfm['Segment'].value_counts()
        colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#22D3EE']
        
        plt.figure(figsize=(8, 8))
        plt.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Customer Segments — SmartOps RFM Analysis')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def train_churn_model(self, rfm: pd.DataFrame):
        """Trains GradientBoostingClassifier on RFM features and saves model + scaler."""
        X = rfm[['Recency', 'Frequency', 'Monetary']]
        y = rfm['Churned']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        print("--- Churn Model Evaluation Report ---")
        print(classification_report(y_test, y_pred))

        # Save artifacts
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.model, self.model_path)
        print(f"✅ Scaler saved to {self.scaler_path}")
        print(f"✅ Model saved to {self.model_path}")
        
        return self.model, self.scaler