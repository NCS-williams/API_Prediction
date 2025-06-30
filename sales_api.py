"""
Sales Prediction API
==================

Flask-based REST API for sales prediction and real-time data processing.
Provides endpoints for data ingestion, predictions, analytics, and monitoring.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlite3
import joblib
from pathlib import Path
import logging
import os
import threading
import queue
from contextlib import contextmanager

# Import our pipeline classes
from realtime_pipeline import RealTimeDataPipeline, DataStreamAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
pipeline = None
data_api = None

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(pipeline.db_path if pipeline else 'sales_realtime.db')
    try:
        yield conn
    finally:
        conn.close()

def init_pipeline():
    """Initialize the pipeline and API"""
    global pipeline, data_api
    pipeline = RealTimeDataPipeline()
    data_api = DataStreamAPI(pipeline)
    logger.info("Pipeline initialized")

# API Routes

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #e9ecef; padding: 2px 5px; }
            h1 { color: #333; }
            h2 { color: #666; }
        </style>
    </head>
    <body>
        <h1>Sales Prediction API</h1>
        <p>Welcome to the Sales Prediction and Analytics API. This API provides real-time sales forecasting, trend analysis, and data ingestion capabilities.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/status</span>
            <p>Check API status and system health</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/data/ingest</span>
            <p>Ingest new sales data point</p>
            <pre>
{
    "datum": "2024-01-01",
    "sales": {
        "M01AB": 5.2,
        "M01AE": 3.1,
        "N02BA": 4.5,
        "N02BE": 35.0,
        "N05B": 12.0,
        "N05C": 1.5,
        "R03": 3.0,
        "R06": 2.0
    }
}
            </pre>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/predictions/latest</span>
            <p>Get latest predictions</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/predictions/future?days=30</span>
            <p>Generate future predictions for specified number of days</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/analytics/trends</span>
            <p>Get trend analysis and seasonality patterns</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/data/recent?days=7</span>
            <p>Get recent sales data</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/alerts</span>
            <p>Get recent alerts and anomalies</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/models/performance</span>
            <p>Get model performance metrics</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="url">/api/simulate</span>
            <p>Simulate real-time data for testing</p>
            <pre>{"days": 7}</pre>
        </div>
        
        <h2>Product Predictions:</h2>
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/predictions/product/{product_code}</span>
            <p>Get predictions for a specific product</p>
        </div>
        
        <h2>Product Forecasts:</h2>
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/api/forecasts/product/{product_code}?days=30</span>
            <p>Generate future forecast for a specific product</p>
        </div>
        
        <h2>Dashboard:</h2>
        <div class="endpoint">
            <span class="method">GET</span> <span class="url">/dashboard</span>
            <p>Interactive dashboard for monitoring and analytics</p>
        </div>
    </body>
    </html>
    """
    return html_template

@app.route('/api/status')
def api_status():
    """Get API status and health check"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipeline_initialized": pipeline is not None,
            "models_loaded": len(pipeline.models) if pipeline else 0,
            "database_accessible": False
        }
        
        # Check database connection
        if pipeline:
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sales_data")
                    data_count = cursor.fetchone()[0]
                    status["database_accessible"] = True
                    status["total_records"] = data_count
            except Exception as e:
                status["database_error"] = str(e)
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Ingest new sales data"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        if 'datum' not in data or 'sales' not in data:
            return jsonify({
                "error": "Missing required fields. Expected format: {'datum': 'YYYY-MM-DD', 'sales': {...}}"
            }), 400
        
        # Process the data
        result = data_api.receive_sales_data(data)
        
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/latest')
def get_latest_predictions():
    """Get latest predictions"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        predictions = data_api.get_latest_predictions()
        return jsonify({
            "predictions": predictions,
            "count": len(predictions) if isinstance(predictions, list) else 0,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/future')
def get_future_predictions():
    """Generate future predictions"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        days = request.args.get('days', 30, type=int)
        if days < 1 or days > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
        
        # Check if we have trained models
        if not pipeline.models:
            return jsonify({"error": "No trained models available. Please train models first."}), 400
        
        # Generate future predictions
        from sales_ml_pipeline import SalesMLPipeline
        ml_pipeline = SalesMLPipeline('salesdaily.xls')
        ml_pipeline.load_and_preprocess_data()
        
        # Use existing models if available
        if hasattr(pipeline, 'models') and pipeline.models:
            # Generate predictions using the real-time pipeline's approach
            future_dates = pd.date_range(
                start=datetime.now().date() + timedelta(days=1),
                periods=days,
                freq='D'
            )
            
            predictions_list = []
            for date in future_dates:
                # Simplified prediction using historical average (fallback)
                with get_db_connection() as conn:
                    recent_avg = pd.read_sql_query('''
                        SELECT AVG(total_sales) as avg_sales FROM sales_data 
                        WHERE datum >= date('now', '-30 days')
                    ''', conn)
                
                avg_sales = recent_avg['avg_sales'].iloc[0] if not recent_avg.empty else 50.0
                
                # Add some seasonal variation
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_factor = 0.8 if date.weekday() in [5, 6] else 1.0
                
                predicted_sales = avg_sales * seasonal_factor * weekend_factor
                
                predictions_list.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_sales": round(predicted_sales, 2),
                    "model_used": "trend_based"
                })
        
        return jsonify({
            "future_predictions": predictions_list,
            "prediction_period_days": days,
            "generated_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error generating future predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/trends')
def get_trends_analysis():
    """Get trend analysis and seasonality"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        with get_db_connection() as conn:
            # Get monthly trends
            monthly_trends = pd.read_sql_query('''
                SELECT 
                    strftime('%Y-%m', datum) as month,
                    AVG(total_sales) as avg_sales,
                    COUNT(*) as days_count
                FROM sales_data 
                GROUP BY strftime('%Y-%m', datum)
                ORDER BY month
            ''', conn)
            
            # Get weekly patterns
            weekly_patterns = pd.read_sql_query('''
                SELECT 
                    strftime('%w', datum) as day_of_week,
                    AVG(total_sales) as avg_sales
                FROM sales_data 
                GROUP BY strftime('%w', datum)
                ORDER BY day_of_week
            ''', conn)
            
            # Get recent trend (last 30 days vs previous 30 days)
            trend_comparison = pd.read_sql_query('''
                SELECT 
                    CASE 
                        WHEN datum >= date('now', '-30 days') THEN 'recent'
                        WHEN datum >= date('now', '-60 days') THEN 'previous'
                    END as period,
                    AVG(total_sales) as avg_sales
                FROM sales_data 
                WHERE datum >= date('now', '-60 days')
                GROUP BY period
            ''', conn)
        
        # Calculate trend direction
        trend_direction = "stable"
        if len(trend_comparison) == 2:
            recent = trend_comparison[trend_comparison['period'] == 'recent']['avg_sales'].iloc[0]
            previous = trend_comparison[trend_comparison['period'] == 'previous']['avg_sales'].iloc[0]
            change_pct = ((recent - previous) / previous) * 100
            
            if change_pct > 5:
                trend_direction = "increasing"
            elif change_pct < -5:
                trend_direction = "decreasing"
        
        # Day of week mapping
        dow_mapping = {
            '0': 'Sunday', '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday',
            '4': 'Thursday', '5': 'Friday', '6': 'Saturday'
        }
        
        weekly_patterns['day_name'] = weekly_patterns['day_of_week'].map(dow_mapping)
        
        return jsonify({
            "trend_analysis": {
                "direction": trend_direction,
                "monthly_trends": monthly_trends.to_dict('records'),
                "weekly_patterns": weekly_patterns.to_dict('records'),
                "peak_day": weekly_patterns.loc[weekly_patterns['avg_sales'].idxmax(), 'day_name'],
                "low_day": weekly_patterns.loc[weekly_patterns['avg_sales'].idxmin(), 'day_name']
            },
            "generated_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in trends analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/recent')
def get_recent_data():
    """Get recent sales data"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        days = request.args.get('days', 7, type=int)
        if days < 1 or days > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
        
        with get_db_connection() as conn:
            recent_data = pd.read_sql_query(f'''
                SELECT * FROM sales_data 
                WHERE datum >= date('now', '-{days} days')
                ORDER BY datum DESC
            ''', conn)
        
        return jsonify({
            "data": recent_data.to_dict('records'),
            "count": len(recent_data),
            "period_days": days,
            "retrieved_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting recent data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        alerts = data_api.get_alerts()
        return jsonify({
            "alerts": alerts,
            "count": len(alerts) if isinstance(alerts, list) else 0,
            "retrieved_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Check if performance file exists
        performance_file = Path('saved_models/Total_Sales_performance.json')
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
            
            return jsonify({
                "model_performance": performance_data,
                "models_available": list(performance_data.keys()),
                "retrieved_at": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "message": "No performance data available. Train models first.",
                "models_loaded": len(pipeline.models) if pipeline.models else 0,
                "retrieved_at": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_data():
    """Simulate real-time data for testing"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        data = request.get_json()
        days = data.get('days', 7) if data else 7
        
        if days < 1 or days > 30:
            return jsonify({"error": "Days must be between 1 and 30"}), 400
        
        # Run simulation in background thread
        def run_simulation():
            pipeline.simulate_realtime_data(num_days=days)
        
        thread = threading.Thread(target=run_simulation)
        thread.start()
        
        return jsonify({
            "message": f"Simulation started for {days} days",
            "status": "running",
            "started_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/product/<product>')
def get_product_predictions(product):
    """Get predictions for a specific product"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Validate product
        valid_products = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06', 'Total_Sales']
        if product not in valid_products:
            return jsonify({
                "error": f"Invalid product. Valid products: {valid_products}"
            }), 400
        
        # Check if we need to generate predictions for existing data
        with get_db_connection() as conn:
            # First check if we have any predictions for this product
            check_query = f"""
                SELECT COUNT(*) as count FROM sales_data 
                WHERE predicted_{product.lower() if product != 'Total_Sales' else 'total_sales'} IS NOT NULL 
                AND predicted_{product.lower() if product != 'Total_Sales' else 'total_sales'} > 0
            """
            
            try:
                result = pd.read_sql_query(check_query, conn)
                prediction_count = result['count'].iloc[0] if not result.empty else 0
                
                # If no predictions exist, generate them for existing data
                if prediction_count == 0:
                    logger.info(f"No predictions found for {product}, generating...")
                    pipeline.add_prediction_for_existing_data(product)
                
                # Now get the predictions
                query = f"""
                    SELECT datum, 
                           {product if product != 'Total_Sales' else 'total_sales'} as actual, 
                           predicted_{product.lower() if product != 'Total_Sales' else 'total_sales'} as predicted
                    FROM sales_data 
                    WHERE predicted_{product.lower() if product != 'Total_Sales' else 'total_sales'} IS NOT NULL
                    AND predicted_{product.lower() if product != 'Total_Sales' else 'total_sales'} > 0
                    ORDER BY datum DESC 
                    LIMIT 30
                """
                
                df = pd.read_sql_query(query, conn)
                predictions = []
                
                for _, row in df.iterrows():
                    predictions.append({
                        "date": row['datum'],
                        "actual": float(row['actual']) if pd.notna(row['actual']) else None,
                        "predicted": float(row['predicted']) if pd.notna(row['predicted']) else None,
                        "product": product
                    })
                
                return jsonify({
                    "product": product,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Database error for {product}: {e}")
                return jsonify({
                    "product": product,
                    "predictions": [],
                    "message": f"Database error: {str(e)}. Try adding some data first.",
                    "timestamp": datetime.now().isoformat()
                })
    
    except Exception as e:
        logger.error(f"Error getting product predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecasts/product/<product>')
def get_product_forecast(product):
    """Generate future forecast for a specific product"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Validate product
        valid_products = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06', 'Total_Sales']
        if product not in valid_products:
            return jsonify({
                "error": f"Invalid product. Valid products: {valid_products}"
            }), 400
        
        days = request.args.get('days', 30, type=int)
        if days < 1 or days > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
        
        # Load ML pipeline and generate forecasts
        from sales_ml_pipeline import SalesMLPipeline
        
        # Check if we have saved models for this product
        model_path = f'models/best_model_{product}.joblib'
        if not os.path.exists(model_path):
            return jsonify({
                "error": f"No trained model found for product {product}. Please train models first.",
                "product": product
            }), 400
        
        ml_pipeline = SalesMLPipeline('salesdaily.xls')
        ml_pipeline.load_and_preprocess_data()
        
        # Load the trained models
        try:
            ml_pipeline.models = {}
            ml_pipeline.scalers = {}
            ml_pipeline.best_models = {}
            
            # Load model and related data
            model_data = joblib.load(model_path)
            ml_pipeline.models[product] = model_data['models']
            ml_pipeline.scalers[product] = model_data['scaler']
            ml_pipeline.best_models[product] = model_data['best_model_info']
            
            # Generate forecast
            forecasts = ml_pipeline.forecast_product(product, days)
            
            return jsonify({
                "product": product,
                "forecasts": forecasts,
                "days_ahead": days,
                "model_used": ml_pipeline.best_models[product]['name'],
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error loading model for {product}: {e}")
            return jsonify({
                "error": f"Failed to load model for product {product}: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Error generating product forecast: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/products')
def get_available_products():
    """Get list of available products"""
    try:
        products = [
            {"code": "M01AB", "name": "Anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives and related substances"},
            {"code": "M01AE", "name": "Anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives"},
            {"code": "N02BA", "name": "Other analgesics and antipyretics, Salicylic acid and derivatives"},
            {"code": "N02BE", "name": "Other analgesics and antipyretics, Pyrazolones and Anilides"},
            {"code": "N05B", "name": "Psycholeptics drugs, Anxiolytic drugs"},
            {"code": "N05C", "name": "Psycholeptics drugs, Hypnotics and sedatives drugs"},
            {"code": "R03", "name": "Drugs for obstructive airway diseases"},
            {"code": "R06", "name": "Antihistamines for systemic use"},
            {"code": "Total_Sales", "name": "Total Sales (Sum of all products)"}
        ]
        
        # Check which products have trained models
        for product in products:
            model_path = f'models/best_model_{product["code"]}.joblib'
            product["has_model"] = os.path.exists(model_path)
        
        return jsonify({
            "products": products,
            "count": len(products),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard for monitoring and analytics"""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Sales Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; margin: 15px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metrics { display: flex; justify-content: space-between; flex-wrap: wrap; }
        .metric { background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; margin: 5px; flex: 1; min-width: 150px; }
        .controls { margin: 20px 0; }
        select, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #3498db; color: white; cursor: pointer; }
        button:hover { background: #2980b9; }
        .chart-container { position: relative; height: 400px; margin: 20px 0; }
        .product-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .loading { text-align: center; color: #666; }
        .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }
        .success { color: #27ae60; background: #d5f4e6; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 Sales Prediction Dashboard</h1>
            <p>Real-time sales forecasting and analytics for pharmaceutical products</p>
        </div>

        <div class="card">
            <h2>📊 System Status</h2>
            <div class="metrics" id="systemMetrics">
                <div class="metric">
                    <h3 id="totalProducts">-</h3>
                    <p>Products</p>
                </div>
                <div class="metric">
                    <h3 id="modelsLoaded">-</h3>
                    <p>Models Loaded</p>
                </div>
                <div class="metric">
                    <h3 id="lastUpdate">-</h3>
                    <p>Last Update</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Individual Product Forecasts</h2>
            <div class="controls">
                <select id="productSelect">
                    <option value="">Select a product...</option>
                </select>
                <select id="daysSelect">
                    <option value="7">7 days</option>
                    <option value="14">14 days</option>
                    <option value="30" selected>30 days</option>
                </select>
                <button onclick="generateForecast()">Generate Forecast</button>
                <button onclick="generateAllForecasts()">Forecast All Products</button>
            </div>
            <div class="chart-container">
                <canvas id="forecastChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>📈 Product Comparison</h2>
            <div class="product-grid" id="productGrid">
                <!-- Product cards will be inserted here -->
            </div>
        </div>

        <div class="card">
            <h2>⚡ Quick Actions</h2>
            <button onclick="addTestData()">Add Test Data</button>
            <button onclick="refreshDashboard()">Refresh Dashboard</button>
            <button onclick="viewAPIStatus()">View API Status</button>
            <div id="actionResult" style="margin-top: 10px;"></div>
        </div>
    </div>

    <script>
        let forecastChart = null;
        let products = [];

        // Initialize dashboard
        async function initDashboard() {
            try {
                await loadProducts();
                await updateSystemMetrics();
                await loadProductGrid();
            } catch (error) {
                console.error('Error initializing dashboard:', error);
            }
        }

        // Load available products
        async function loadProducts() {
            try {
                const response = await fetch('/api/products');
                const data = await response.json();
                products = data.products;
                
                const select = document.getElementById('productSelect');
                select.innerHTML = '<option value="">Select a product...</option>';
                
                products.forEach(product => {
                    const option = document.createElement('option');
                    option.value = product.code;
                    option.textContent = `${product.code} - ${product.name.substring(0, 50)}...`;
                    select.appendChild(option);
                });
                
                document.getElementById('totalProducts').textContent = products.length;
                document.getElementById('modelsLoaded').textContent = products.filter(p => p.has_model).length;
            } catch (error) {
                console.error('Error loading products:', error);
            }
        }

        // Update system metrics
        async function updateSystemMetrics() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }

        // Generate forecast for selected product
        async function generateForecast() {
            const productCode = document.getElementById('productSelect').value;
            const days = document.getElementById('daysSelect').value;
            
            if (!productCode) {
                alert('Please select a product');
                return;
            }

            try {
                const response = await fetch(`/api/forecasts/product/${productCode}?days=${days}`);
                const data = await response.json();
                
                if (response.ok) {
                    displayForecastChart(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error generating forecast:', error);
                alert('Error generating forecast');
            }
        }

        // Display forecast chart
        function displayForecastChart(data) {
            const ctx = document.getElementById('forecastChart').getContext('2d');
            
            if (forecastChart) {
                forecastChart.destroy();
            }
            
            const labels = data.forecasts.map(f => f.date);
            const values = data.forecasts.map(f => f.predicted_value);
            
            forecastChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: `${data.product} Forecast (${data.model_used})`,
                        data: values,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Sales Volume'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `Sales Forecast for ${data.product}`
                        }
                    }
                }
            });
        }

        // Generate forecasts for all products
        async function generateAllForecasts() {
            const days = document.getElementById('daysSelect').value;
            const grid = document.getElementById('productGrid');
            grid.innerHTML = '<div class="loading">Generating forecasts for all products...</div>';
            
            const forecasts = [];
            
            for (const product of products.filter(p => p.has_model)) {
                try {
                    const response = await fetch(`/api/forecasts/product/${product.code}?days=${days}`);
                    const data = await response.json();
                    if (response.ok) {
                        forecasts.push(data);
                    }
                } catch (error) {
                    console.error(`Error forecasting ${product.code}:`, error);
                }
            }
            
            displayProductGrid(forecasts);
        }

        // Load product grid
        async function loadProductGrid() {
            const grid = document.getElementById('productGrid');
            grid.innerHTML = '<div class="loading">Loading product information...</div>';
            
            let html = '';
            for (const product of products) {
                const hasModel = product.has_model ? '✅' : '❌';
                html += `
                    <div class="card">
                        <h3>${hasModel} ${product.code}</h3>
                        <p>${product.name}</p>
                        <p><strong>Model Available:</strong> ${product.has_model ? 'Yes' : 'No'}</p>
                        ${product.has_model ? `
                            <button onclick="quickForecast('${product.code}')">Quick Forecast</button>
                        ` : ''}
                    </div>
                `;
            }
            grid.innerHTML = html;
        }

        // Display product grid with forecasts
        function displayProductGrid(forecasts) {
            const grid = document.getElementById('productGrid');
            let html = '';
            
            forecasts.forEach(forecast => {
                const avgValue = forecast.forecasts.reduce((sum, f) => sum + f.predicted_value, 0) / forecast.forecasts.length;
                const trend = forecast.forecasts.length > 1 ? 
                    (forecast.forecasts[forecast.forecasts.length - 1].predicted_value > forecast.forecasts[0].predicted_value ? '📈' : '📉') : '📊';
                
                html += `
                    <div class="card">
                        <h3>${trend} ${forecast.product}</h3>
                        <p><strong>Model:</strong> ${forecast.model_used}</p>
                        <p><strong>Avg Forecast:</strong> ${avgValue.toFixed(2)}</p>
                        <p><strong>Period:</strong> ${forecast.days_ahead} days</p>
                        <button onclick="selectProduct('${forecast.product}')">View Details</button>
                    </div>
                `;
            });
            
            grid.innerHTML = html;
        }

        // Quick forecast for a product
        async function quickForecast(productCode) {
            document.getElementById('productSelect').value = productCode;
            await generateForecast();
        }

        // Select product
        function selectProduct(productCode) {
            document.getElementById('productSelect').value = productCode;
            generateForecast();
        }

        // Add test data
        async function addTestData() {
            const testData = {
                datum: new Date().toISOString().split('T')[0],
                sales: {
                    M01AB: Math.random() * 10,
                    M01AE: Math.random() * 8,
                    N02BA: Math.random() * 6,
                    N02BE: Math.random() * 50,
                    N05B: Math.random() * 20,
                    N05C: Math.random() * 5,
                    R03: Math.random() * 15,
                    R06: Math.random() * 8
                }
            };
            
            try {
                const response = await fetch('/api/data/ingest', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(testData)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('actionResult');
                
                if (response.ok) {
                    resultDiv.innerHTML = '<div class="success">✅ Test data added successfully!</div>';
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ Error: ${result.error}</div>`;
                }
            } catch (error) {
                document.getElementById('actionResult').innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            }
        }

        // Refresh dashboard
        async function refreshDashboard() {
            document.getElementById('actionResult').innerHTML = '<div class="loading">Refreshing...</div>';
            await initDashboard();
            document.getElementById('actionResult').innerHTML = '<div class="success">✅ Dashboard refreshed!</div>';
        }

        // View API status
        async function viewAPIStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                alert(`API Status: ${JSON.stringify(data, null, 2)}`);
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Auto-refresh every 30 seconds
        setInterval(updateSystemMetrics, 30000);

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
    """
    return html_template

@app.get('/api/medecins')
def get_medecins():
    """Get list of medecins"""
    try:
        medecins = DataStreamAPI.get_medecins(self=pipeline)
        return jsonify({
            "medecins": medecins,
            "count": len(medecins) if isinstance(medecins, list) else 0,
            "retrieved_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting medecins: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server"""
    init_pipeline()
    logger.info(f"Starting Sales Prediction API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server(debug=True)
