#!/usr/bin/env python3
"""
Train Individual Product Models
==============================

Quick script to train ML models for each product individually.
"""

from sales_ml_pipeline import SalesMLPipeline
import os

def main():
    print("=== Training Models for All Products ===")
    
    # Initialize pipeline
    pipeline = SalesMLPipeline('salesdaily.xls')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = pipeline.load_and_preprocess_data()
    
    # Train models for all products
    print("Training models for all products...")
    best_models = pipeline.train_all_product_models()
    
    # Create neural networks for key products
    key_products = ['Total_Sales', 'M01AB', 'N02BE', 'N05B']  # Train neural networks for most important products
    for product in key_products:
        if product in pipeline.models:
            print(f"Training neural network for {product}...")
            try:
                pipeline.create_neural_network(product)
            except Exception as e:
                print(f"Warning: Could not train neural network for {product}: {e}")
    
    # Save all models
    print("Saving all trained models...")
    pipeline.save_all_models()
    
    # Print summary
    print("\n=== Training Summary ===")
    for product, model_info in best_models.items():
        print(f"{product}: {model_info['name']} (R² = {model_info['performance']['r2']:.3f})")
    
    # Check saved models
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        print(f"\n=== Saved Models ({len(model_files)} files) ===")
        for file in sorted(model_files):
            print(f"✓ {file}")
    
    print("\n✅ All product models trained and saved!")
    print("You can now use the API endpoints:")
    print("- GET /api/products (list all products)")
    print("- GET /api/predictions/product/<product_code>")
    print("- GET /api/forecasts/product/<product_code>?days=30")

if __name__ == "__main__":
    main()
