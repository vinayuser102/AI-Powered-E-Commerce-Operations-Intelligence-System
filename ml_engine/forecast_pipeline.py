import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

class DemandForecastEngine:
    def __init__(self, model_dir: str = "models", output_dir: str = "outputs"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model_path = os.path.join(model_dir, "demand_prophet_model.pkl")
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def prepare_time_series(self, df_master: pd.DataFrame) -> pd.DataFrame:
        """Aggregates master order data into daily order counts for Prophet."""
        df = df_master.copy()
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        daily_orders = df.groupby(
            df['order_purchase_timestamp'].dt.date
        )['order_id'].nunique().reset_index()

        daily_orders.columns = ['ds', 'y']
        daily_orders['ds'] = pd.to_datetime(daily_orders['ds'])
        return daily_orders

    def train_and_forecast(self, daily_orders: pd.DataFrame, horizon_days: int = 90):
        """Splits time series (80/20), trains Prophet, evaluates MAE, generates future forecast, and saves model."""
        split = int(len(daily_orders) * 0.8)
        train = daily_orders[:split]
        test = daily_orders[split:]

        print(f"Training on {len(train)} days | Testing on {len(test)} days")

        # Train model on 80% split
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False
        )
        model.fit(train)

        # Forecast full range + horizon
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Calculate accuracy metrics against test set
        test_forecast = forecast[forecast['ds'].isin(test['ds'])][['ds', 'yhat']]
        merged = test.merge(test_forecast, on='ds')

        mae = mean_absolute_error(merged['y'], merged['yhat'])
        accuracy = 100 - (mae / merged['y'].mean() * 100)

        print(f"MAE: {mae:.2f} orders/day")
        print(f"Forecast Accuracy: {accuracy:.1f}%")

        # Save trained Prophet model
        joblib.dump(model, self.model_path)
        print(f"✅ Prophet model saved to {self.model_path}")

        return model, forecast

    def save_forecast_plot(self, model: Prophet, forecast: pd.DataFrame, file_path: str = None):
        """Plots and exports the 90-day demand forecast figure."""
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'demand_forecast.png')

        fig = model.plot(forecast)
        plt.title('SmartOps — 90-Day Demand Forecast')
        plt.xlabel('Date')
        plt.ylabel('Orders per Day')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        print(f"✅ Forecast plot saved to {file_path}")