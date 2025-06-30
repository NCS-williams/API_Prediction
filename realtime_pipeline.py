"""
Real-time Sales Data Pipeline
============================

This module handles real-time data ingestion, processing, and prediction updates.
It can connect to various data sources and continuously update predictions.
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import threading
import queue
import joblib
import sqlite3
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataPipeline:
    """
    Real-time data pipeline for continuous sales prediction updates
    """
    
    def __init__(self, model_path='saved_models', db_path='sales_realtime.db'):
        """Initialize the real-time pipeline"""
        self.model_path = Path(model_path)
        self.db_path = db_path
        self.data_queue = queue.Queue()
        self.is_running = False
        self.models = {}
        self.scalers = {}
        self.product_columns = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
        
        # Initialize database
        self.init_database()
        
        # Load trained models
        self.load_models()
    
    def init_database(self):
        """Initialize SQLite database for storing real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                datum DATE,
                M01AB REAL,
                M01AE REAL,
                N02BA REAL,
                N02BE REAL,
                N05B REAL,
                N05C REAL,
                R03 REAL,
                R06 REAL,
                total_sales REAL,
                predicted_M01AB REAL,
                predicted_M01AE REAL,
                predicted_N02BA REAL,
                predicted_N02BE REAL,
                predicted_N05B REAL,
                predicted_N05C REAL,
                predicted_R03 REAL,
                predicted_R06 REAL,
                predicted_total_sales REAL,
                source TEXT DEFAULT 'realtime'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prediction_date DATE,
                target_product TEXT,
                predicted_value REAL,
                model_used TEXT,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                message TEXT,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_models(self):
        """Load pre-trained models for all products"""
        models_dir = Path('models')
        if not models_dir.exists():
            logger.warning(f"Models directory {models_dir} does not exist")
            return
        
        try:
            # Load models for all products
            products = ['Total_Sales'] + self.product_columns
            
            for product in products:
                model_file = models_dir / f'best_model_{product}.joblib'
                if model_file.exists():
                    model_data = joblib.load(model_file)
                    self.models[product] = model_data
                    logger.info(f"Loaded models for {product}")
                else:
                    logger.warning(f"No model file found for {product}")
            
            logger.info(f"Loaded {len(self.models)} product models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def add_data_point(self, data_point):
        """Add a new data point to the processing queue"""
        """
        data_point format:
        {
            'datum': '2024-01-01',
            'M01AB': 5.2,
            'M01AE': 3.1,
            ... (other product sales)
        }
        """
        self.data_queue.put(data_point)
        logger.info(f"Added data point to queue: {data_point['datum']}")
    
    def process_data_point(self, data_point):
        """Process a single data point and store in database"""
        try:
            # Calculate total sales
            total_sales = sum(data_point.get(col, 0) for col in self.product_columns)
            
            # Generate predictions for all products
            predictions = self.generate_all_predictions(data_point)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sales_data (
                    datum, M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06, total_sales,
                    predicted_M01AB, predicted_M01AE, predicted_N02BA, predicted_N02BE, 
                    predicted_N05B, predicted_N05C, predicted_R03, predicted_R06, predicted_total_sales
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_point['datum'],
                data_point.get('M01AB', 0),
                data_point.get('M01AE', 0),
                data_point.get('N02BA', 0),
                data_point.get('N02BE', 0),
                data_point.get('N05B', 0),
                data_point.get('N05C', 0),
                data_point.get('R03', 0),
                data_point.get('R06', 0),
                total_sales,
                predictions.get('M01AB', 0),
                predictions.get('M01AE', 0),
                predictions.get('N02BA', 0),
                predictions.get('N02BE', 0),
                predictions.get('N05B', 0),
                predictions.get('N05C', 0),
                predictions.get('R03', 0),
                predictions.get('R06', 0),
                predictions.get('Total_Sales', 0)
            ))
            
            conn.commit()
            conn.close()
            
            # Check for anomalies
            self.check_anomalies(data_point, total_sales)
            
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
    
    def generate_all_predictions(self, data_point):
        """Generate predictions for all products based on current data"""
        predictions = {}
        
        if not self.models:
            logger.warning("No models loaded for prediction")
            return predictions
        
        try:
            # Load the ML pipeline to use proper feature engineering
            from sales_ml_pipeline import SalesMLPipeline
            ml_pipeline = SalesMLPipeline('salesdaily.xls')
            ml_pipeline.load_and_preprocess_data()
            
            # Get the last row of processed data for feature engineering
            last_row = ml_pipeline.processed_data.tail(1).copy()
            
            # Create a new data point with current sales data
            current_date = pd.to_datetime(data_point['datum'])
            new_row = pd.Series({
                'datum': current_date,
                'Year': current_date.year,
                'Month': current_date.month,
                'DayOfWeek': current_date.dayofweek,
                'DayOfYear': current_date.dayofyear,
                'WeekOfYear': current_date.isocalendar().week,
                'Quarter': current_date.quarter,
                'IsWeekend': int(current_date.weekday() in [5, 6])
            })
            
            # Add current product sales
            for product in self.product_columns:
                if hasattr(data_point, 'get'):
                    new_row[product] = data_point.get(product, 0)
                elif isinstance(data_point, dict):
                    # Handle nested sales dict
                    if 'sales' in data_point:
                        new_row[product] = data_point['sales'].get(product, 0)
                    else:
                        new_row[product] = data_point.get(product, 0)
                else:
                    new_row[product] = 0
            
            # Calculate total sales
            new_row['Total_Sales'] = sum(new_row[col] for col in self.product_columns)
            
            # Add lag features using historical data
            for product in self.product_columns:
                for lag in [1, 3, 7, 14, 30]:
                    if len(ml_pipeline.processed_data) >= lag:
                        new_row[f'{product}_lag_{lag}'] = ml_pipeline.processed_data[product].iloc[-lag]
                    else:
                        new_row[f'{product}_lag_{lag}'] = ml_pipeline.processed_data[product].mean()
            
            # Add rolling averages using historical data
            for product in self.product_columns:
                for window in [3, 7, 14, 30]:
                    if len(ml_pipeline.processed_data) >= window:
                        new_row[f'{product}_rolling_{window}'] = ml_pipeline.processed_data[product].tail(window).mean()
                    else:
                        new_row[f'{product}_rolling_{window}'] = ml_pipeline.processed_data[product].mean()
            
            # For each product, generate prediction using the trained models
            for product in ['Total_Sales'] + self.product_columns:
                if product not in self.models:
                    logger.warning(f"No model loaded for {product}")
                    continue
                
                try:
                    model_data = self.models[product]
                    best_model_info = model_data['best_model_info']
                    best_model_name = best_model_info['name']
                    best_model = model_data['models'][best_model_name]['model']
                    scaler = model_data['scaler']
                    
                    # Create features using the same logic as training
                    feature_cols = [
                        'Year', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Quarter', 'IsWeekend'
                    ]
                    
                    # Add lag features
                    for prod in self.product_columns:
                        for lag in [1, 3, 7, 14, 30]:
                            feature_cols.append(f'{prod}_lag_{lag}')
                    
                    # Add rolling averages
                    for prod in self.product_columns:
                        for window in [3, 7, 14, 30]:
                            feature_cols.append(f'{prod}_rolling_{window}')
                    
                    # Add other product features for individual product prediction
                    if product != 'Total_Sales' and product in self.product_columns:
                        other_products = [p for p in self.product_columns if p != product]
                        feature_cols.extend(other_products)
                    
                    # Extract features
                    feature_values = []
                    for col in feature_cols:
                        if col in new_row:
                            feature_values.append(new_row[col])
                        else:
                            # Use zero for missing features
                            feature_values.append(0)
                            logger.warning(f"Missing feature {col} for {product}, using 0")
                    
                    X_pred = np.array(feature_values).reshape(1, -1)
                    
                    # Make prediction
                    if best_model_name in ['Ridge', 'ElasticNet']:
                        # Use scaled features for linear models
                        X_pred_scaled = scaler.transform(X_pred)
                        prediction = best_model.predict(X_pred_scaled)[0]
                    else:
                        # Use original features for tree-based models
                        prediction = best_model.predict(X_pred)[0]
                    
                    # Ensure prediction is non-negative
                    prediction = max(0, prediction)
                    predictions[product] = round(prediction, 2)
                    
                    logger.info(f"Generated prediction for {product}: {prediction:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error generating prediction for {product}: {e}")
                    predictions[product] = 0
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in generate_all_predictions: {e}")
            return predictions
    
    def update_predictions(self, new_data_point):
        """Update predictions based on new data"""
        if not self.models:
            logger.warning("No models loaded for prediction")
            return
        
        try:
            # Get recent data for feature engineering
            conn = sqlite3.connect(self.db_path)
            recent_data = pd.read_sql_query('''
                SELECT * FROM sales_data 
                ORDER BY datum DESC 
                LIMIT 30
            ''', conn)
            conn.close()
            
            if len(recent_data) == 0:
                logger.warning("No recent data available for prediction")
                return
            
            # Create features for prediction
            features = self.create_prediction_features(new_data_point, recent_data)
            
            # Make predictions with each model
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    if model_name in ['Ridge', 'ElasticNet'] and 'Total_Sales' in self.scalers:
                        # Use scaled features for linear models
                        features_scaled = self.scalers['Total_Sales'].transform([features])
                        pred = model.predict(features_scaled)[0]
                    else:
                        # Use original features for tree-based models
                        pred = model.predict([features])[0]
                    
                    predictions[model_name] = max(0, pred)  # Ensure non-negative
                except Exception as e:
                    logger.error(f"Error making prediction with {model_name}: {e}")
            
            if predictions:
                # Store predictions
                self.store_predictions(predictions, new_data_point['datum'])
                logger.info(f"Updated predictions: {predictions}")
        
        except Exception as e:
            logger.error(f"Error updating predictions: {e}")
    
    def create_prediction_features(self, new_data_point, recent_data):
        """Create features for prediction from new data point and recent history"""
        date = pd.to_datetime(new_data_point['datum'])
        
        features = []
        
        # Time-based features
        features.extend([
            date.year,          # Year
            date.month,         # Month
            date.dayofweek,     # DayOfWeek
            date.dayofyear,     # DayOfYear
            date.isocalendar().week,  # WeekOfYear
            date.quarter,       # Quarter
            int(date.dayofweek in [5, 6])  # IsWeekend
        ])
        
        # Lag features (simplified - using recent averages)
        for product in self.product_columns:
            product_col = product.lower()  # Database columns are lowercase
            if product_col in recent_data.columns:
                product_data = recent_data[product_col].dropna()
                if not product_data.empty:
                    # Lag features (1, 3, 7, 14, 30 days)
                    for lag in [1, 3, 7, 14, 30]:
                        if len(product_data) >= lag:
                            features.append(product_data.iloc[-(lag)])
                        else:
                            features.append(product_data.mean())
                    
                    # Rolling averages (3, 7, 14, 30 days)
                    for window in [3, 7, 14, 30]:
                        if len(product_data) >= window:
                            features.append(product_data.tail(window).mean())
                        else:
                            features.append(product_data.mean())
                else:
                    # Default values if no data
                    features.extend([0] * 8)  # 4 lag + 4 rolling
            else:
                features.extend([0] * 8)
        
        return features
    
    def store_predictions(self, predictions, prediction_date):
        """Store predictions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for model_name, pred_value in predictions.items():
            cursor.execute('''
                INSERT INTO predictions (prediction_date, target_product, predicted_value, model_used)
                VALUES (?, ?, ?, ?)
            ''', (prediction_date, 'Total_Sales', pred_value, model_name))
        
        conn.commit()
        conn.close()
    
    def check_anomalies(self, data_point, total_sales):
        """Check for anomalies in the new data point"""
        try:
            # Get recent data for comparison
            conn = sqlite3.connect(self.db_path)
            recent_data = pd.read_sql_query('''
                SELECT total_sales FROM sales_data 
                WHERE datum >= date('now', '-30 days')
                ORDER BY datum DESC
            ''', conn)
            conn.close()
            
            if len(recent_data) < 5:
                return  # Not enough data for anomaly detection
            
            # Calculate statistics
            mean_sales = recent_data['total_sales'].mean()
            std_sales = recent_data['total_sales'].std()
            
            # Check for anomalies (values outside 2 standard deviations)
            if abs(total_sales - mean_sales) > 2 * std_sales:
                alert_type = "high_sales" if total_sales > mean_sales else "low_sales"
                message = f"Anomaly detected: Total sales {total_sales:.2f} is significantly {'higher' if total_sales > mean_sales else 'lower'} than recent average {mean_sales:.2f}"
                severity = "high" if abs(total_sales - mean_sales) > 3 * std_sales else "medium"
                
                self.create_alert(alert_type, message, severity)
        
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    def create_alert(self, alert_type, message, severity):
        """Create an alert and store in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, message, severity)
            VALUES (?, ?, ?)
        ''', (alert_type, message, severity))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"ALERT ({severity}): {message}")
    
    def start_pipeline(self, poll_interval=1):
        """Start the real-time pipeline"""
        self.is_running = True
        logger.info("Real-time pipeline started")
        
        while self.is_running:
            try:
                # Process any data in the queue
                while not self.data_queue.empty():
                    data_point = self.data_queue.get()
                    self.process_data_point(data_point)
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in pipeline: {e}")
                time.sleep(poll_interval)
        
        self.is_running = False
    
    def stop_pipeline(self):
        """Stop the real-time pipeline"""
        self.is_running = False
        logger.info("Pipeline stopped")
    
    def get_recent_predictions(self, days=7):
        """Get recent predictions from database"""
        conn = sqlite3.connect(self.db_path)
        predictions = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        conn.close()
        
        return predictions
    
    def get_recent_alerts(self, days=7):
        """Get recent alerts from database"""
        conn = sqlite3.connect(self.db_path)
        alerts = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        conn.close()
        
        return alerts
    
    def simulate_realtime_data(self, num_days=7):
        """Simulate real-time data for testing (generates synthetic data)"""
        logger.info(f"Starting simulation of {num_days} days of data...")
        
        # Get the last date from database or use current date
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(datum) FROM sales_data')
        result = cursor.fetchone()
        conn.close()
        
        if result[0]:
            start_date = pd.to_datetime(result[0]) + timedelta(days=1)
        else:
            start_date = datetime.now().date()
        
        # Generate synthetic data
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            
            # Generate realistic sales data with some randomness
            base_sales = {
                'M01AB': np.random.normal(5, 2),
                'M01AE': np.random.normal(3, 1.5),
                'N02BA': np.random.normal(4, 1.8),
                'N02BE': np.random.normal(35, 10),
                'N05B': np.random.normal(12, 4),
                'N05C': np.random.normal(1, 0.5),
                'R03': np.random.normal(3, 2),
                'R06': np.random.normal(2, 1)
            }
            
            # Ensure non-negative values
            for key in base_sales:
                base_sales[key] = max(0, base_sales[key])
            
            # Add some weekly seasonality (lower sales on weekends)
            if current_date.weekday() in [5, 6]:  # Weekend
                for key in base_sales:
                    base_sales[key] *= 0.8
            
            data_point = {
                'datum': current_date.strftime('%Y-%m-%d'),
                **base_sales
            }
            
            self.add_data_point(data_point)
            
            # Process immediately for simulation
            if not self.data_queue.empty():
                point = self.data_queue.get()
                self.process_data_point(point)
            
            # Small delay to simulate real-time
            time.sleep(0.1)
        
        logger.info(f"Simulation complete: {num_days} days of data generated")
    
    def add_prediction_for_existing_data(self, product):
        """Generate predictions for existing data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all data that doesn't have predictions for this product
            query = f"""
                SELECT * FROM sales_data 
                WHERE predicted_{product.lower()} IS NULL OR predicted_{product.lower()} = 0
                ORDER BY datum DESC 
                LIMIT 50
            """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) == 0:
                logger.info(f"No data without predictions for {product}")
                conn.close()
                return
            
            logger.info(f"Generating predictions for {len(df)} existing records for {product}")
            
            for _, row in df.iterrows():
                # Prepare data point in the expected format
                data_point = {
                    'datum': row['datum'],
                    'sales': {
                        'M01AB': row['M01AB'],
                        'M01AE': row['M01AE'],
                        'N02BA': row['N02BA'],
                        'N02BE': row['N02BE'],
                        'N05B': row['N05B'],
                        'N05C': row['N05C'],
                        'R03': row['R03'],
                        'R06': row['R06']
                    }
                }
                
                # Generate predictions
                predictions = self.generate_all_predictions(data_point)
                
                # Update the database with predictions
                if predictions:
                    update_query = f"""
                        UPDATE sales_data 
                        SET predicted_{product.lower()} = ?
                        WHERE id = ?
                    """
                    cursor = conn.cursor()
                    cursor.execute(update_query, (predictions.get(product, 0), row['id']))
                    conn.commit()
            
            conn.close()
            logger.info(f"Completed prediction generation for {product}")
            
        except Exception as e:
            logger.error(f"Error adding predictions for existing data: {e}")
    
class DataStreamAPI:
    """
    API interface for the real-time data pipeline
    """
    
    def __init__(self, pipeline):
        """Initialize with pipeline instance"""
        self.pipeline = pipeline
        
        self.product_columns = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
        
    def get_medecins(self):
        return self.product_columns
        
    def get_latest_predictions(self):
        """Get latest predictions from database"""
        try:
            conn = sqlite3.connect(self.pipeline.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM sales_data 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', conn)
            conn.close()
            
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting latest predictions: {e}")
            return []
    
    def get_alerts(self):
        """Get recent alerts"""
        try:
            conn = sqlite3.connect(self.pipeline.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT 20
            ''', conn)
            conn.close()
            
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

# Main execution
if __name__ == "__main__":
    # Test the pipeline
    pipeline = RealTimeDataPipeline()
    
    # Test data point
    test_data = {
        'datum': '2024-01-01',
        'sales': {
            'M01AB': 5.2,
            'M01AE': 3.1,
            'N02BA': 4.5,
            'N02BE': 35.0,
            'N05B': 12.0,
            'N05C': 1.5,
            'R03': 8.0,
            'R06': 2.0
        }
    }
    
    pipeline.add_data_point(test_data)
    pipeline.process_data_point(test_data)
    
    print("Real-time pipeline test completed!")
