"""
Complete Sales ML System Demo
============================

This script demonstrates the complete sales ML system including:
1. Training ML models on historical data
2. Setting up real-time pipeline
3. Starting the API server
4. Testing with sample data

Run this script to see the complete system in action.
"""

import sys
import os
import time
import threading
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def run_ml_training():
    """Run the ML pipeline to train models"""
    print("=== Step 1: Training ML Models ===")
    
    try:
        from sales_ml_pipeline import SalesMLPipeline
        
        # Initialize and run the ML pipeline
        pipeline = SalesMLPipeline('salesdaily.xls')
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = pipeline.load_and_preprocess_data()
        
        # Train models for all products
        print("Training machine learning models for all products...")
        pipeline.train_all_product_models()
        
        # Create neural networks for all products
        print("Training neural networks for all products...")
        products = ['Total_Sales'] + pipeline.product_columns
        for product in products:
            print(f"Training neural network for {product}...")
            pipeline.create_neural_network(product)
        
        # Analyze trends
        print("Analyzing trends and seasonality...")
        pipeline.detect_trends_and_seasonality()
        
        # Save all models
        print("Saving all trained models...")
        pipeline.save_all_models()
        
        # Generate predictions
        print("Generating future predictions...")
        pipeline.predict_future_sales(days_ahead=30)
        
        # Create visualizations
        print("Creating visualizations...")
        pipeline.visualize_results('Total_Sales')
        
        # Generate report
        print("Generating analysis report...")
        pipeline.generate_report('Total_Sales')
        
        print("✓ ML training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error in ML training: {e}")
        return False

def start_api_server():
    """Start the API server in a separate thread"""
    print("=== Step 2: Starting API Server ===")
    
    try:
        from sales_api import run_server
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, kwargs={
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False
        })
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        # Test server connection
        response = requests.get('http://localhost:5000/api/status', timeout=5)
        if response.status_code == 200:
            print("✓ API server started successfully!")
            print("  - API Documentation: http://localhost:5000")
            print("  - Dashboard: http://localhost:5000/dashboard")
            print("  - API Status: http://localhost:5000/api/status")
            return True
        else:
            print("✗ API server started but not responding correctly")
            return False
            
    except Exception as e:
        print(f"✗ Error starting API server: {e}")
        return False

def test_api_endpoints():
    """Test the API endpoints with sample data"""
    print("=== Step 3: Testing API Endpoints ===")
    
    base_url = 'http://localhost:5000/api'
    
    try:
        # Test 1: Check API status
        print("Testing API status...")
        response = requests.get(f'{base_url}/status')
        if response.status_code == 200:
            print("✓ API status check passed")
            status = response.json()
            print(f"  Pipeline initialized: {status.get('pipeline_initialized')}")
            print(f"  Models loaded: {status.get('models_loaded')}")
        else:
            print("✗ API status check failed")
            return False
        
        # Test 2: Simulate some data
        print("Simulating real-time data...")
        sim_response = requests.post(f'{base_url}/simulate', 
                                   json={'days': 5},
                                   timeout=10)
        if sim_response.status_code == 200:
            print("✓ Data simulation started")
        else:
            print("✗ Data simulation failed")
        
        # Wait for simulation to process
        time.sleep(2)
        
        # Test 3: Ingest a sample data point
        print("Testing data ingestion...")
        sample_data = {
            'datum': (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'sales': {
                'M01AB': 5.2,
                'M01AE': 3.1,
                'N02BA': 4.5,
                'N02BE': 35.0,
                'N05B': 12.0,
                'N05C': 1.5,
                'R03': 3.0,
                'R06': 2.0
            }
        }
        
        response = requests.post(f'{base_url}/data/ingest', json=sample_data)
        if response.status_code == 200:
            print("✓ Data ingestion test passed")
        else:
            print("✗ Data ingestion test failed")
            print(f"  Response: {response.text}")
        
        # Test 4: Get recent data
        print("Testing recent data retrieval...")
        response = requests.get(f'{base_url}/data/recent?days=7')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved {data.get('count', 0)} recent records")
        else:
            print("✗ Recent data retrieval failed")
        
        # Test 5: Get predictions
        print("Testing predictions retrieval...")
        response = requests.get(f'{base_url}/predictions/latest')
        if response.status_code == 200:
            predictions = response.json()
            print(f"✓ Retrieved {predictions.get('count', 0)} predictions")
        else:
            print("✗ Predictions retrieval failed")
        
        # Test 6: Get future predictions
        print("Testing future predictions...")
        response = requests.get(f'{base_url}/predictions/future?days=7')
        if response.status_code == 200:
            future_preds = response.json()
            print(f"✓ Generated {len(future_preds.get('future_predictions', []))} future predictions")
        else:
            print("✗ Future predictions failed")
        
        # Test 7: Get trends analysis
        print("Testing trends analysis...")
        response = requests.get(f'{base_url}/analytics/trends')
        if response.status_code == 200:
            trends = response.json()
            print("✓ Trends analysis completed")
            trend_direction = trends.get('trend_analysis', {}).get('direction', 'unknown')
            print(f"  Trend direction: {trend_direction}")
        else:
            print("✗ Trends analysis failed")
        
        # Test 8: Check alerts
        print("Testing alerts retrieval...")
        response = requests.get(f'{base_url}/alerts')
        if response.status_code == 200:
            alerts = response.json()
            print(f"✓ Retrieved {alerts.get('count', 0)} alerts")
        else:
            print("✗ Alerts retrieval failed")
        
        print("✓ All API tests completed!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API connection error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing API endpoints: {e}")
        return False

def demonstrate_realtime_usage():
    """Demonstrate real-time usage patterns"""
    print("=== Step 4: Demonstrating Real-time Usage ===")
    
    base_url = 'http://localhost:5000/api'
    
    try:
        print("Simulating continuous data ingestion...")
        
        # Simulate receiving data from multiple sources
        for i in range(5):
            # Generate realistic sales data
            current_date = datetime.now().date() + timedelta(days=i+2)
            
            # Simulate different scenarios
            scenarios = [
                "normal_day",
                "high_sales_day", 
                "low_sales_day",
                "weekend_day",
                "anomaly_day"
            ]
            
            scenario = scenarios[i % len(scenarios)]
            
            # Base sales values
            base_sales = {
                'M01AB': np.random.normal(5, 1),
                'M01AE': np.random.normal(3, 0.8),
                'N02BA': np.random.normal(4, 1.2),
                'N02BE': np.random.normal(35, 8),
                'N05B': np.random.normal(12, 3),
                'N05C': np.random.normal(1.5, 0.5),
                'R03': np.random.normal(3, 1.5),
                'R06': np.random.normal(2, 0.8)
            }
            
            # Apply scenario modifications
            if scenario == "high_sales_day":
                base_sales = {k: v * 1.5 for k, v in base_sales.items()}
            elif scenario == "low_sales_day":
                base_sales = {k: v * 0.6 for k, v in base_sales.items()}
            elif scenario == "weekend_day":
                base_sales = {k: v * 0.8 for k, v in base_sales.items()}
            elif scenario == "anomaly_day":
                # Create an anomaly in one product
                base_sales['N02BE'] *= 3.0
            
            # Ensure non-negative values
            for key in base_sales:
                base_sales[key] = max(0, base_sales[key])
            
            data_point = {
                'datum': current_date.strftime('%Y-%m-%d'),
                'sales': base_sales
            }
            
            print(f"  Ingesting data for {current_date} ({scenario})...")
            response = requests.post(f'{base_url}/data/ingest', json=data_point)
            
            if response.status_code == 200:
                print(f"    ✓ Data ingested successfully")
            else:
                print(f"    ✗ Failed to ingest data: {response.text}")
            
            # Small delay to simulate real-time
            time.sleep(0.5)
        
        # Check for any alerts generated
        print("Checking for generated alerts...")
        response = requests.get(f'{base_url}/alerts')
        if response.status_code == 200:
            alerts = response.json()
            if alerts.get('count', 0) > 0:
                print(f"  Generated {alerts['count']} alerts!")
                for alert in alerts['alerts'][:3]:  # Show first 3 alerts
                    print(f"    - {alert['severity'].upper()}: {alert['message']}")
            else:
                print("  No alerts generated")
        
        print("✓ Real-time demonstration completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error in real-time demonstration: {e}")
        return False

def show_usage_examples():
    """Show examples of how to use the system"""
    print("=== Usage Examples ===")
    
    examples = {
        "Ingest Sales Data": {
            "method": "POST",
            "url": "http://localhost:5000/api/data/ingest",
            "body": {
                "datum": "2024-01-15",
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
        },
        "Get Future Predictions": {
            "method": "GET",
            "url": "http://localhost:5000/api/predictions/future?days=30"
        },
        "Get Trend Analysis": {
            "method": "GET",
            "url": "http://localhost:5000/api/analytics/trends"
        },
        "Get Recent Alerts": {
            "method": "GET",
            "url": "http://localhost:5000/api/alerts"
        }
    }
    
    print("API Usage Examples:")
    print("==================")
    
    for title, example in examples.items():
        print(f"\n{title}:")
        print(f"  {example['method']} {example['url']}")
        if 'body' in example:
            print(f"  Body: {json.dumps(example['body'], indent=2)}")
    
    print("\nDashboard Access:")
    print("================")
    print("Interactive Dashboard: http://localhost:5000/dashboard")
    print("API Documentation: http://localhost:5000")
    
    print("\nPython Client Example:")
    print("=====================")
    print("""
import requests

# Ingest new sales data
data = {
    'datum': '2024-01-15',
    'sales': {
        'M01AB': 5.2, 'M01AE': 3.1, 'N02BA': 4.5, 'N02BE': 35.0,
        'N05B': 12.0, 'N05C': 1.5, 'R03': 3.0, 'R06': 2.0
    }
}
response = requests.post('http://localhost:5000/api/data/ingest', json=data)

# Get predictions
predictions = requests.get('http://localhost:5000/api/predictions/future?days=7')
print(predictions.json())
    """)

def main():
    """Main demo function"""
    print("=" * 60)
    print("   COMPLETE SALES ML SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    
    success_steps = []
    
    # Step 1: Train ML models
    if run_ml_training():
        success_steps.append("ML Training")
    
    # Step 2: Start API server
    if start_api_server():
        success_steps.append("API Server")
    
    # Step 3: Test API endpoints
    if test_api_endpoints():
        success_steps.append("API Testing")
    
    # Step 4: Demonstrate real-time usage
    if demonstrate_realtime_usage():
        success_steps.append("Real-time Demo")
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print("\n" + "=" * 60)
    print("   DEMONSTRATION SUMMARY")
    print("=" * 60)
    print(f"Completed steps: {len(success_steps)}/4")
    for step in success_steps:
        print(f"  ✓ {step}")
    
    if len(success_steps) == 4:
        print("\n🎉 Complete system demonstration successful!")
        print("\nThe sales prediction system is now running and ready to use:")
        print("  • ML models trained and saved")
        print("  • Real-time pipeline active")
        print("  • API server running on http://localhost:5000")
        print("  • Dashboard available at http://localhost:5000/dashboard")
        print("\nThe server will continue running. Press Ctrl+C to stop.")
        
        # Keep the demo running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nDemo stopped by user. Goodbye!")
    else:
        print("\n⚠️  Some steps failed. Check the error messages above.")
        print("You may need to install additional dependencies or fix configuration issues.")

if __name__ == "__main__":
    main()
