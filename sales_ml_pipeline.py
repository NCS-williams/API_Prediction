"""
Sales Prediction and Trend Detection ML Pipeline
==============================================

This module provides a comprehensive machine learning solution for:
1. Sales forecasting using multiple algorithms
2. Trend detection and seasonality analysis
3. Real-time data pipeline capabilities
4. Model evaluation and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Time series libraries
from datetime import datetime, timedelta
import joblib
import optuna
import json
import os

class SalesMLPipeline:
    """
    Comprehensive ML Pipeline for Sales Prediction and Trend Analysis
    """
    
    def __init__(self, data_path='salesdaily.xls'):
        """Initialize the pipeline with data path"""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.model_performance = {}
        
        # Product columns (sales data)
        self.product_columns = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
        
    def get_medecins(self):
        return self.product_columns
        
    def load_and_preprocess_data(self):
        """Load and preprocess the sales data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Convert date column
        self.data['datum'] = pd.to_datetime(self.data['datum'], format='%m/%d/%Y')
        self.data = self.data.sort_values('datum').reset_index(drop=True)
        
        # Create additional time features
        self.data['DayOfWeek'] = self.data['datum'].dt.dayofweek
        self.data['DayOfYear'] = self.data['datum'].dt.dayofyear
        self.data['WeekOfYear'] = self.data['datum'].dt.isocalendar().week
        self.data['Quarter'] = self.data['datum'].dt.quarter
        self.data['IsWeekend'] = self.data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Create lag features for each product
        for product in self.product_columns:
            for lag in [1, 3, 7, 14, 30]:
                self.data[f'{product}_lag_{lag}'] = self.data[product].shift(lag)
        
        # Create rolling averages
        for product in self.product_columns:
            for window in [3, 7, 14, 30]:
                self.data[f'{product}_rolling_{window}'] = self.data[product].rolling(window=window).mean()
        
        # Create total sales column
        self.data['Total_Sales'] = self.data[self.product_columns].sum(axis=1)
        
        # Remove rows with NaN values (due to lag features)
        self.processed_data = self.data.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.processed_data)} records")
        print(f"Date range: {self.processed_data['datum'].min()} to {self.processed_data['datum'].max()}")
        
        return self.processed_data
    
    def create_features(self, target_product=None):
        """Create feature matrix for ML models"""
        if target_product is None:
            target_product = 'Total_Sales'
            
        # Feature columns
        feature_cols = [
            'Year', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Quarter', 'IsWeekend'
        ]
        
        # Add lag features
        lag_cols = [col for col in self.processed_data.columns if '_lag_' in col]
        rolling_cols = [col for col in self.processed_data.columns if '_rolling_' in col]
        
        feature_cols.extend(lag_cols)
        feature_cols.extend(rolling_cols)
        
        # Add other product sales as features (for individual product prediction)
        if target_product != 'Total_Sales' and target_product in self.product_columns:
            other_products = [p for p in self.product_columns if p != target_product]
            feature_cols.extend(other_products)
        
        X = self.processed_data[feature_cols]
        y = self.processed_data[target_product]
        
        return X, y, feature_cols
    
    def train_models(self, target_product='Total_Sales', test_size=0.2):
        """Train multiple ML models"""
        print(f"Training models for {target_product}...")
        
        X, y, feature_cols = self.create_features(target_product)
        
        # Time series split to maintain temporal order
        tscv = TimeSeriesSplit(n_splits=5)
        split_idx = list(tscv.split(X))[-1]  # Use the last split
        train_idx, test_idx = split_idx
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_product] = scaler
        
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Ridge', 'ElasticNet']:
                # Use scaled features for linear models
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        self.models[target_product] = results
        self.model_performance[target_product] = {
            name: {'mae': res['mae'], 'rmse': res['rmse'], 'r2': res['r2']} 
            for name, res in results.items()
        }
        
        # Store test data for visualization
        self.test_data = {
            'y_test': y_test,
            'X_test': X_test,
            'dates': self.processed_data.iloc[test_idx]['datum']
        }
        
        return results
    
    def create_neural_network(self, target_product='Total_Sales'):
        """Create and train a neural network model"""
        print(f"Training Neural Network for {target_product}...")
        
        X, y, feature_cols = self.create_features(target_product)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        split_idx = list(tscv.split(X))[-1]
        train_idx, test_idx = split_idx
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        if target_product not in self.scalers:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[target_product] = scaler
        else:
            scaler = self.scalers[target_product]
            X_train_scaled = scaler.transform(X_train)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Create neural network
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Add to models
        if target_product not in self.models:
            self.models[target_product] = {}
        
        self.models[target_product]['NeuralNetwork'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'history': history
        }
        
        print(f"Neural Network - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        return model, history
    
    def detect_trends_and_seasonality(self):
        """Analyze trends and seasonal patterns"""
        print("Analyzing trends and seasonality...")
        
        trends_analysis = {}
        
        for product in self.product_columns + ['Total_Sales']:
            if product not in self.processed_data.columns:
                continue
                
            product_data = self.processed_data[['datum', product]].copy()
            product_data.set_index('datum', inplace=True)
            
            # Monthly aggregation
            monthly_sales = product_data.resample('M').sum()
            
            # Trend analysis (linear regression on time)
            X_time = np.arange(len(monthly_sales)).reshape(-1, 1)
            y_sales = monthly_sales[product].values
            
            from sklearn.linear_model import LinearRegression
            trend_model = LinearRegression()
            trend_model.fit(X_time, y_sales)
            trend_slope = trend_model.coef_[0]
            
            # Seasonality analysis
            product_data['month'] = product_data.index.month
            monthly_avg = product_data.groupby('month')[product].mean()
            
            # Detect seasonal patterns
            seasonal_strength = monthly_avg.std() / monthly_avg.mean()
            
            trends_analysis[product] = {
                'trend_slope': trend_slope,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'seasonal_strength': seasonal_strength,
                'monthly_pattern': monthly_avg.to_dict(),
                'peak_month': monthly_avg.idxmax(),
                'low_month': monthly_avg.idxmin()
            }
        
        self.trends_analysis = trends_analysis
        return trends_analysis
    
    def predict_future_sales(self, days_ahead=30, target_product='Total_Sales'):
        """Predict future sales"""
        print(f"Predicting sales for next {days_ahead} days...")
        
        if target_product not in self.models:
            raise ValueError(f"Model for {target_product} not trained yet!")
        
        # Get the best performing model
        best_model_name = min(
            self.models[target_product].keys(),
            key=lambda x: self.models[target_product][x]['rmse']
        )
        best_model = self.models[target_product][best_model_name]['model']
        
        print(f"Using {best_model_name} for prediction (best RMSE)")
        
        # Create future dates
        last_date = self.processed_data['datum'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Create feature template for future predictions
        latest_data = self.processed_data.tail(30).copy()  # Get last 30 days for lag features
        
        predictions = []
        
        for future_date in future_dates:
            # Create features for this future date
            future_features = {
                'Year': future_date.year,
                'Month': future_date.month,
                'DayOfWeek': future_date.dayofweek,
                'DayOfYear': future_date.dayofyear,
                'WeekOfYear': future_date.isocalendar().week,
                'Quarter': future_date.quarter,
                'IsWeekend': int(future_date.dayofweek in [5, 6])
            }
            
            # Add lag features (use recent actual data and predictions)
            for product in self.product_columns:
                for lag in [1, 3, 7, 14, 30]:
                    if len(latest_data) >= lag:
                        future_features[f'{product}_lag_{lag}'] = latest_data[product].iloc[-lag]
                    else:
                        future_features[f'{product}_lag_{lag}'] = latest_data[product].mean()
            
            # Add rolling averages
            for product in self.product_columns:
                for window in [3, 7, 14, 30]:
                    if len(latest_data) >= window:
                        future_features[f'{product}_rolling_{window}'] = latest_data[product].tail(window).mean()
                    else:
                        future_features[f'{product}_rolling_{window}'] = latest_data[product].mean()
            
            # Add other product features if predicting individual product
            if target_product != 'Total_Sales' and target_product in self.product_columns:
                other_products = [p for p in self.product_columns if p != target_product]
                for product in other_products:
                    future_features[product] = latest_data[product].mean()
            
            # Create feature vector
            X, _, feature_cols = self.create_features(target_product)
            feature_vector = pd.DataFrame([future_features])[feature_cols]
            
            # Make prediction
            if best_model_name in ['Ridge', 'ElasticNet']:
                scaler = self.scalers[target_product]
                feature_vector_scaled = scaler.transform(feature_vector)
                pred = best_model.predict(feature_vector_scaled)[0]
            else:
                pred = best_model.predict(feature_vector)[0]
            
            predictions.append(max(0, pred))  # Ensure non-negative predictions
            
            # Update latest_data with prediction (simplified approach)
            if target_product == 'Total_Sales':
                # Distribute total sales among products based on historical ratios
                product_ratios = self.processed_data[self.product_columns].mean() / self.processed_data['Total_Sales'].mean()
                new_row = {col: 0 for col in latest_data.columns}
                new_row['datum'] = future_date
                for i, product in enumerate(self.product_columns):
                    new_row[product] = pred * product_ratios[product]
                new_row['Total_Sales'] = pred
            else:
                new_row = latest_data.iloc[-1].copy()
                new_row['datum'] = future_date
                new_row[target_product] = pred
            
            latest_data = pd.concat([latest_data, pd.DataFrame([new_row])]).tail(30)
        
        future_predictions = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions,
            'target_product': target_product,
            'model_used': best_model_name
        })
        
        self.future_predictions = future_predictions
        return future_predictions
    
    def visualize_results(self, target_product='Total_Sales'):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        if target_product not in self.models:
            print(f"No models trained for {target_product}")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Model Performance Comparison',
                'Actual vs Predicted (Best Model)',
                'Sales Trend Over Time',
                'Seasonal Patterns',
                'Feature Importance',
                'Future Predictions'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Model Performance Comparison
        model_names = list(self.models[target_product].keys())
        rmse_values = [self.models[target_product][name]['rmse'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_values, name='RMSE'),
            row=1, col=1
        )
        
        # 2. Actual vs Predicted (Best Model)
        best_model_name = min(model_names, key=lambda x: self.models[target_product][x]['rmse'])
        y_test = self.test_data['y_test']
        y_pred = self.models[target_product][best_model_name]['predictions']
        
        fig.add_trace(
            go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                name=f'{best_model_name} Predictions',
                text=[f'Actual: {a:.1f}<br>Predicted: {p:.1f}' for a, p in zip(y_test, y_pred)]
            ),
            row=1, col=2
        )
        
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=2
        )
        
        # 3. Sales Trend Over Time
        historical_data = self.processed_data.tail(365)  # Last year
        fig.add_trace(
            go.Scatter(
                x=historical_data['datum'],
                y=historical_data[target_product],
                mode='lines',
                name=f'{target_product} Trend'
            ),
            row=2, col=1
        )
        
        # 4. Seasonal Patterns
        if hasattr(self, 'trends_analysis') and target_product in self.trends_analysis:
            monthly_pattern = self.trends_analysis[target_product]['monthly_pattern']
            months = list(monthly_pattern.keys())
            values = list(monthly_pattern.values())
            
            fig.add_trace(
                go.Bar(x=months, y=values, name='Monthly Average'),
                row=2, col=2
            )
        
        # 5. Feature Importance (for tree-based models)
        if best_model_name in ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting']:
            model = self.models[target_product][best_model_name]['model']
            X, _, feature_cols = self.create_features(target_product)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(10)
                
                fig.add_trace(
                    go.Bar(
                        x=feature_imp['importance'],
                        y=feature_imp['feature'],
                        orientation='h',
                        name='Feature Importance'
                    ),
                    row=3, col=1
                )
        
        # 6. Future Predictions
        if hasattr(self, 'future_predictions'):
            fig.add_trace(
                go.Scatter(
                    x=self.future_predictions['date'],
                    y=self.future_predictions['predicted_sales'],
                    mode='lines+markers',
                    name='Future Predictions'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Sales ML Pipeline Results - {target_product}",
            showlegend=True
        )
        
        fig.show()
        
        # Save the plot
        fig.write_html(f'sales_analysis_{target_product.lower()}.html')
        print(f"Visualizations saved as sales_analysis_{target_product.lower()}.html")
    
    def save_models(self, target_product='Total_Sales'):
        """Save trained models and scalers"""
        if target_product not in self.models:
            print(f"No models to save for {target_product}")
            return
        
        os.makedirs('saved_models', exist_ok=True)
        
        for model_name, model_data in self.models[target_product].items():
            if model_name == 'NeuralNetwork':
                # Save TensorFlow model
                model_data['model'].save(f'saved_models/{target_product}_{model_name}')
            else:
                # Save sklearn models
                joblib.dump(
                    model_data['model'],
                    f'saved_models/{target_product}_{model_name}.joblib'
                )
        
        # Save scalers
        joblib.dump(
            self.scalers[target_product],
            f'saved_models/{target_product}_scaler.joblib'
        )
        
        # Save performance metrics
        with open(f'saved_models/{target_product}_performance.json', 'w') as f:
            json.dump(self.model_performance[target_product], f, indent=2)
        
        print(f"Models saved for {target_product}")
    
    def save_all_models(self):
        """Save all trained models for all products"""
        print("Saving all trained models...")
        
        os.makedirs('models', exist_ok=True)
        
        products = ['Total_Sales'] + self.product_columns
        for product in products:
            if product in self.models and product in self.best_models:
                print(f"Saving models for {product}...")
                
                # Save the complete model data for each product
                model_data = {
                    'models': self.models[product],
                    'scaler': self.scalers[product],
                    'best_model_info': self.best_models[product],
                    'model_performance': self.model_performance[product],
                    'product': product,
                    'saved_at': datetime.now().isoformat()
                }
                
                # Save to a single file per product
                joblib.dump(model_data, f'models/best_model_{product}.joblib')
                
                # Also save the best model separately for easy loading
                best_model_name = self.best_models[product]['name']
                best_model = self.models[product][best_model_name]['model']
                
                if best_model_name == 'NeuralNetwork':
                    # Save TensorFlow model
                    best_model.save(f'models/{product}_neural_network.keras')
                else:
                    # Save sklearn models
                    joblib.dump(best_model, f'models/{product}_best.joblib')
        
        print(f"All models saved to 'models/' directory")

    def generate_report(self, target_product='Total_Sales'):
        """Generate a comprehensive analysis report"""
        report = f"""
        Sales Prediction and Trend Analysis Report
        =========================================
        
        Target Product: {target_product}
        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Data Overview:
        - Total Records: {len(self.processed_data)}
        - Date Range: {self.processed_data['datum'].min()} to {self.processed_data['datum'].max()}
        - Products Analyzed: {', '.join(self.product_columns)}
        
        Model Performance:
        """
        
        if target_product in self.model_performance:
            for model_name, metrics in self.model_performance[target_product].items():
                report += f"""
        {model_name}:
          - MAE: {metrics['mae']:.2f}
          - RMSE: {metrics['rmse']:.2f}
          - R²: {metrics['r2']:.3f}"""
        
        if hasattr(self, 'trends_analysis') and target_product in self.trends_analysis:
            trends = self.trends_analysis[target_product]
            report += f"""
        
        Trend Analysis:
        - Trend Direction: {trends['trend_direction'].title()}
        - Trend Slope: {trends['trend_slope']:.2f} units/month
        - Seasonal Strength: {trends['seasonal_strength']:.2f}
        - Peak Sales Month: {trends['peak_month']}
        - Lowest Sales Month: {trends['low_month']}
        """
        
        if hasattr(self, 'future_predictions'):
            avg_prediction = self.future_predictions['predicted_sales'].mean()
            report += f"""
        
        Future Predictions:
        - Prediction Period: {len(self.future_predictions)} days
        - Average Predicted Sales: {avg_prediction:.2f}
        - Model Used: {self.future_predictions['model_used'].iloc[0]}
        """
        
        # Save report
        with open(f'sales_analysis_report_{target_product.lower()}.txt', 'w') as f:
            f.write(report)
        
        print("Analysis report generated!")
        print(report)
        
        return report

    def train_all_product_models(self, test_size=0.2):
        """Train models for each product individually and total sales"""
        print("Training models for all products...")
        
        # Train model for total sales
        self.train_models('Total_Sales', test_size)
        
        # Train models for each individual product
        for product in self.product_columns:
            print(f"\nTraining models for product: {product}")
            self.train_models(product, test_size)
        
        # Find best model for each product
        self.best_models = {}
        for product in ['Total_Sales'] + self.product_columns:
            if product in self.model_performance:
                best_model_name = max(
                    self.model_performance[product].keys(),
                    key=lambda k: self.model_performance[product][k]['r2']
                )
                self.best_models[product] = {
                    'name': best_model_name,
                    'performance': self.model_performance[product][best_model_name]
                }
                print(f"Best model for {product}: {best_model_name} (R² = {self.model_performance[product][best_model_name]['r2']:.3f})")
        
        return self.best_models

    def predict_single_product(self, product, last_n_days=30):
        """Make predictions for a single product"""
        if product not in self.models:
            raise ValueError(f"No trained model found for product: {product}")
        
        # Get the best model for this product
        best_model_name = self.best_models[product]['name']
        best_model = self.models[product][best_model_name]['model']
        
        # Get recent data for prediction
        recent_data = self.processed_data.tail(last_n_days).copy()
        
        # Create features for prediction
        X, _, feature_cols = self.create_features(product)
        X_recent = X.tail(last_n_days)
        
        # Make predictions
        if best_model_name in ['Ridge', 'ElasticNet']:
            # Use scaled features for linear models
            scaler = self.scalers[product]
            X_scaled = scaler.transform(X_recent)
            predictions = best_model.predict(X_scaled)
        else:
            # Use original features for tree-based models
            predictions = best_model.predict(X_recent)
        
        return predictions, recent_data['datum'].values

    def forecast_product(self, product, days_ahead=30):
        """Generate future forecasts for a specific product"""
        if product not in self.models:
            raise ValueError(f"No trained model found for product: {product}")
        
        # Get the best model for this product
        best_model_name = self.best_models[product]['name']
        
        # Generate future dates
        last_date = self.processed_data['datum'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        forecasts = []
        current_data = self.processed_data.copy()
        
        for i, future_date in enumerate(future_dates):
            # Create a new row for prediction
            new_row = pd.Series({
                'datum': future_date,
                'Year': future_date.year,
                'Month': future_date.month,
                'DayOfWeek': future_date.dayofweek,
                'DayOfYear': future_date.dayofyear,
                'WeekOfYear': future_date.isocalendar().week,
                'Quarter': future_date.quarter,
                'IsWeekend': int(future_date.weekday() in [5, 6])
            })
            
            # Add lag features using most recent available data
            for lag in [1, 3, 7, 14, 30]:
                if len(current_data) >= lag:
                    new_row[f'{product}_lag_{lag}'] = current_data[product].iloc[-lag]
                else:
                    new_row[f'{product}_lag_{lag}'] = current_data[product].mean()
            
            # Add rolling averages
            for window in [3, 7, 14, 30]:
                if len(current_data) >= window:
                    new_row[f'{product}_rolling_{window}'] = current_data[product].tail(window).mean()
                else:
                    new_row[f'{product}_rolling_{window}'] = current_data[product].mean()
            
            # Add other product features (use latest values)
            if product != 'Total_Sales':
                for other_product in self.product_columns:
                    if other_product != product:
                        new_row[other_product] = current_data[other_product].iloc[-1]
            
            # Prepare features for prediction
            X, _, feature_cols = self.create_features(product)
            feature_values = []
            
            for col in feature_cols:
                if col in new_row:
                    feature_values.append(new_row[col])
                else:
                    # Use mean for missing features
                    feature_values.append(current_data[col].mean() if col in current_data.columns else 0)
            
            X_pred = np.array(feature_values).reshape(1, -1)
            
            # Make prediction
            best_model = self.models[product][best_model_name]['model']
            if best_model_name in ['Ridge', 'ElasticNet']:
                scaler = self.scalers[product]
                X_pred_scaled = scaler.transform(X_pred)
                prediction = best_model.predict(X_pred_scaled)[0]
            else:
                prediction = best_model.predict(X_pred)[0]
            
            # Ensure prediction is non-negative
            prediction = max(0, prediction)
            
            forecasts.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_value': round(prediction, 2),
                'product': product,
                'model_used': best_model_name
            })
            
            # Add this prediction to current_data for next iteration
            new_row[product] = prediction
            if product != 'Total_Sales':
                # Update total sales
                new_row['Total_Sales'] = sum(new_row.get(p, 0) for p in self.product_columns)
            
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        return forecasts

def main():
    """Main execution function"""
    print("=== Sales ML Pipeline Demo ===")
    
    # Initialize pipeline
    pipeline = SalesMLPipeline('salesdaily.xls')
    
    # Load and preprocess data
    data = pipeline.load_and_preprocess_data()
    
    # Train models for total sales
    print("\n=== Training Models for Total Sales ===")
    pipeline.train_models('Total_Sales')
    
    # Create neural network
    pipeline.create_neural_network('Total_Sales')
    
    # Detect trends and seasonality
    print("\n=== Analyzing Trends and Seasonality ===")
    trends = pipeline.detect_trends_and_seasonality()
    
    # Predict future sales
    print("\n=== Predicting Future Sales ===")
    future_preds = pipeline.predict_future_sales(days_ahead=30)
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    pipeline.visualize_results('Total_Sales')
    
    # Save models
    print("\n=== Saving Models ===")
    pipeline.save_models('Total_Sales')
    
    # Generate report
    print("\n=== Generating Report ===")
    pipeline.generate_report('Total_Sales')
    
    print("\n=== Pipeline Complete ===")
    print("Check the generated files:")
    print("- sales_analysis_total_sales.html (Interactive visualizations)")
    print("- sales_analysis_report_total_sales.txt (Analysis report)")
    print("- saved_models/ directory (Trained models)")

if __name__ == "__main__":
    main()
