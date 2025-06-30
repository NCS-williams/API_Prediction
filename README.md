# Sales Prediction ML System

A comprehensive machine learning system for sales forecasting, trend analysis, and real-time data processing with REST API capabilities.

## Features

### 🤖 Machine Learning
- Multiple ML algorithms (Random Forest, XGBoost, LightGBM, Neural Networks, etc.)
- Automated model comparison and selection
- Feature engineering with lag features and rolling averages
- Time series cross-validation
- Model performance tracking

### 📈 Analytics & Forecasting
- Sales trend detection and seasonality analysis
- Future sales predictions (configurable time horizon)
- Anomaly detection with automated alerts
- Statistical analysis and pattern recognition

### 🚀 Real-time Pipeline
- Real-time data ingestion and processing
- Continuous model predictions updates
- SQLite database for data storage
- Automated anomaly detection and alerting

### 🌐 REST API
- Complete REST API for all functionality
- Real-time data ingestion endpoints
- Prediction and analytics APIs
- Interactive web dashboard
- CORS support for web applications

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Demo
```bash
python demo.py
```

This will:
- Train ML models on your historical data
- Start the API server
- Test all endpoints
- Demonstrate real-time data ingestion
- Launch the interactive dashboard

### 3. Access the System
- **API Documentation**: http://localhost:5000
- **Interactive Dashboard**: http://localhost:5000/dashboard
- **API Status**: http://localhost:5000/api/status

## How It Works

This system analyzes your sales data and predicts future sales using machine learning.

### Simple Workflow

1. **Load Data**: Reads your sales data from the Excel file
2. **Train Models**: Automatically trains 7 different AI models on your data
3. **Make Predictions**: Uses the best model to forecast future sales
4. **Detect Trends**: Identifies patterns and unusual changes in sales
5. **Real-time Updates**: Continuously processes new data as it arrives
6. **Web Dashboard**: Provides an easy-to-use interface to view results

### What You Get

- **Sales Forecasts**: Predict sales for next days, weeks, or months
- **Trend Analysis**: See if sales are going up, down, or staying stable
- **Anomaly Alerts**: Get notified when something unusual happens
- **Interactive Dashboard**: View charts and graphs in your web browser
- **REST API**: Connect to other systems and applications

### Data Flow

```
Your Excel File → AI Training → Predictions → Web Dashboard
                      ↓              ↑
                  Saved Models → Real-time Data → Alerts
```

The system is designed to be simple to use but powerful enough for production environments.

### Individual Product Predictions

The system now supports predictions for each pharmaceutical product individually:

#### Available Products:
- **M01AB**: Anti-inflammatory products (Acetic acid derivatives)
- **M01AE**: Anti-inflammatory products (Propionic acid derivatives)  
- **N02BA**: Analgesics (Salicylic acid derivatives)
- **N02BE**: Analgesics (Pyrazolones and Anilides)
- **N05B**: Anxiolytic drugs
- **N05C**: Hypnotics and sedatives
- **R03**: Drugs for obstructive airway diseases
- **R06**: Antihistamines
- **Total_Sales**: Combined sales of all products

#### API Endpoints:
- `GET /api/products` - List all available products and their model status
- `GET /api/predictions/product/<product_code>` - Get recent predictions for a specific product
- `GET /api/forecasts/product/<product_code>?days=30` - Generate future forecasts for a product

#### Example Usage:
```bash
# Get available products
curl "http://localhost:5000/api/products"

# Get 30-day forecast for M01AB
curl "http://localhost:5000/api/forecasts/product/M01AB?days=30"

# Get recent predictions for N02BE
curl "http://localhost:5000/api/predictions/product/N02BE"
```
