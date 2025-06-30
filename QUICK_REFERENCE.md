# Sales Prediction System - Quick Reference

## 🚀 System Status
- **Data Loaded**: 2,076 sales records (2014-2019)
- **Models Trained**: 6+ ML algorithms
- **Best Model**: Ridge Regression (R² = 0.675)
- **API Status**: Ready to start
- **Real-time Pipeline**: Active

## 📊 Model Performance Summary
```
Model               MAE     RMSE    R²
----------------   -----   -----   -----
Ridge Regression    9.76   13.54   0.675  ⭐ Best
Gradient Boosting  10.67   14.67   0.618
LightGBM           10.87   14.96   0.603
Random Forest      11.51   15.99   0.547
ElasticNet         11.57   16.09   0.541
XGBoost            12.34   17.28   0.471
Neural Network     [Training...]
```

## 🌐 API Endpoints
```
Base URL: http://localhost:5000

System Health:
GET /api/status

Data Management:
POST /api/data/ingest          - Add new sales data
GET /api/data/recent?days=7    - Get recent data

Predictions:
GET /api/predictions/latest    - Get latest predictions
GET /api/predictions/future?days=30 - Generate forecasts

Analytics:
GET /api/analytics/trends      - Trend analysis
GET /api/alerts               - Recent alerts

Testing:
POST /api/simulate            - Simulate test data
```

## 💻 Usage Examples

### Ingest Sales Data
```bash
curl -X POST http://localhost:5000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "datum": "2025-06-30",
    "sales": {
      "M01AB": 5.2, "M01AE": 3.1, "N02BA": 4.5, "N02BE": 35.0,
      "N05B": 12.0, "N05C": 1.5, "R03": 3.0, "R06": 2.0
    }
  }'
```

### Get Future Predictions
```bash
curl http://localhost:5000/api/predictions/future?days=7
```

### Check System Status
```bash
curl http://localhost:5000/api/status
```

## 🐍 Python Client
```python
import requests

# Connect to API
base_url = 'http://localhost:5000/api'

# Add new sales data
data = {
    'datum': '2025-06-30',
    'sales': {
        'M01AB': 5.2, 'M01AE': 3.1, 'N02BA': 4.5, 'N02BE': 35.0,
        'N05B': 12.0, 'N05C': 1.5, 'R03': 3.0, 'R06': 2.0
    }
}
response = requests.post(f'{base_url}/data/ingest', json=data)
print(response.json())

# Get predictions
predictions = requests.get(f'{base_url}/predictions/future?days=30')
print(predictions.json())

# Analyze trends
trends = requests.get(f'{base_url}/analytics/trends')
print(trends.json())
```

## 🖥️ Web Interface
- **Dashboard**: http://localhost:5000/dashboard
- **API Docs**: http://localhost:5000
- **Status**: http://localhost:5000/api/status

## 📁 Generated Files
```
sales_analysis_total_sales.html    - Interactive visualizations
sales_analysis_report_total_sales.txt - Analysis report
saved_models/                      - Trained ML models
  ├── Total_Sales_Ridge.joblib
  ├── Total_Sales_RandomForest.joblib
  ├── Total_Sales_XGBoost.joblib
  ├── Total_Sales_LightGBM.joblib
  ├── Total_Sales_scaler.joblib
  └── Total_Sales_performance.json
sales_realtime.db                  - SQLite database
```

## 🎯 Product Categories
- **M01AB**: Anti-inflammatory products
- **M01AE**: Propionic acid derivatives  
- **N02BA**: Salicylic acid and derivatives
- **N02BE**: Pyrazolone derivatives
- **N05B**: Anxiolytics
- **N05C**: Hypnotics and sedatives
- **R03**: Drugs for obstructive airway diseases
- **R06**: Antihistamines for systemic use

## 🔧 Key Features
- ✅ **Real-time Data Ingestion**
- ✅ **Multiple ML Algorithms**
- ✅ **Automated Model Selection**
- ✅ **Trend Detection & Seasonality**
- ✅ **Anomaly Detection**
- ✅ **REST API with CORS**
- ✅ **Interactive Dashboard**
- ✅ **SQLite Database**
- ✅ **Model Persistence**
- ✅ **Performance Monitoring**

## 🚨 Next Steps
1. **Start the API**: `python demo.py` (currently running)
2. **Access Dashboard**: Open http://localhost:5000/dashboard
3. **Test API**: Use curl commands or Python client
4. **Monitor Performance**: Check alerts and metrics
5. **Integrate Data Sources**: Connect your real data feeds

## 📞 Troubleshooting
- **Port Issues**: Ensure port 5000 is available
- **Model Errors**: Check saved_models/ directory exists
- **Database Issues**: Verify SQLite permissions
- **Import Errors**: Run `pip install -r requirements.txt`
