#!/usr/bin/env python3
"""
Utility script to fix NumPy compatibility issues with saved models
This script helps resolve the 'MT19937 is not a known BitGenerator module' error
"""

import os
import sys
import joblib
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_numpy_version():
    """Check current NumPy version"""
    logger.info(f"Current NumPy version: {np.__version__}")
    return np.__version__

def fix_model_compatibility(model_path, backup=True):
    """Fix NumPy compatibility for a single model file"""
    try:
        # Create backup if requested
        if backup and os.path.exists(model_path):
            backup_path = f"{model_path}.backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(model_path, backup_path)
                logger.info(f"✅ Created backup: {backup_path}")
        
        # Try to load the model
        logger.info(f"🔄 Loading model: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"✅ Successfully loaded: {model_path}")
        
        # Save the model with current NumPy version
        logger.info(f"💾 Re-saving model with current NumPy version: {model_path}")
        joblib.dump(model, model_path)
        logger.info(f"✅ Successfully re-saved: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing {model_path}: {str(e)}")
        return False

def fix_all_models():
    """Fix NumPy compatibility for all model files"""
    model_dir = Path("models/news/models")
    
    if not model_dir.exists():
        logger.error(f"❌ Model directory not found: {model_dir}")
        logger.info("💡 Please ensure your models are in the correct location")
        return False
    
    logger.info(f"🔍 Scanning for models in: {model_dir}")
    
    # List of model files to process
    model_files = [
        model_dir / "spike_model.pkl",
        model_dir / "vectorizer.pkl",
        model_dir / "label_encoders.pkl"
    ]
    
    # Add price models
    price_models_dir = model_dir / "Price_Models"
    if price_models_dir.exists():
        for model_file in price_models_dir.glob("price_model_*.pkl"):
            model_files.append(model_file)
    
    logger.info(f"📋 Found {len(model_files)} model files to process")
    
    # Process each model file
    success_count = 0
    for model_file in model_files:
        if model_file.exists():
            if fix_model_compatibility(str(model_file)):
                success_count += 1
        else:
            logger.warning(f"⚠️ Model file not found: {model_file}")
    
    logger.info(f"🎯 Successfully processed {success_count}/{len(model_files)} model files")
    return success_count > 0

def create_mock_models():
    """Create mock models for testing (when real models are not available)"""
    logger.info("🔧 Creating mock models for testing...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    # Create model directory
    model_dir = Path("models/news/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Price_Models subdirectory
    price_models_dir = model_dir / "Price_Models"
    price_models_dir.mkdir(exist_ok=True)
    
    # Mock spike model
    spike_model = RandomForestClassifier(n_estimators=10, random_state=42)
    spike_model.fit(np.random.rand(100, 10), np.random.randint(0, 2, 100))
    joblib.dump(spike_model, model_dir / "spike_model.pkl")
    logger.info("✅ Created mock spike model")
    
    # Mock vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    sample_texts = ["sample text " + str(i) for i in range(50)]
    vectorizer.fit(sample_texts)
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    logger.info("✅ Created mock vectorizer")
    
    # Mock label encoders
    label_encoders = {}
    hotel_codes = ["AHUN.N0000", "TRAN.N0000", "STAF.N0000"]
    for hotel in hotel_codes:
        le = LabelEncoder()
        le.fit(["up", "down", "neutral"])
        label_encoders[hotel] = le
    
    joblib.dump(label_encoders, model_dir / "label_encoders.pkl")
    logger.info("✅ Created mock label encoders")
    
    # Mock price models
    for hotel in hotel_codes:
        price_model = RandomForestClassifier(n_estimators=10, random_state=42)
        price_model.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
        joblib.dump(price_model, price_models_dir / f"price_model_{hotel}.pkl")
    
    logger.info(f"✅ Created {len(hotel_codes)} mock price models")
    logger.info("🎉 Mock models created successfully!")
    logger.info("💡 These models are for testing only and will not provide accurate predictions")

def main():
    """Main function"""
    print("🔧 NumPy Compatibility Fix Tool")
    print("=" * 50)
    
    # Check NumPy version
    numpy_version = check_numpy_version()
    print(f"📊 NumPy Version: {numpy_version}")
    
    # Check if models exist
    model_dir = Path("models/news/models")
    if not model_dir.exists():
        print("\n❌ No models found!")
        print("💡 Options:")
        print("   1. Copy your trained models to models/news/models/")
        print("   2. Create mock models for testing")
        
        choice = input("\nWould you like to create mock models for testing? (y/n): ").lower()
        if choice == 'y':
            create_mock_models()
            return
        else:
            print("👋 Exiting...")
            return
    
    print(f"\n🔍 Found model directory: {model_dir}")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Fix NumPy compatibility for existing models")
    print("2. Create mock models for testing")
    print("3. Check model status only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🔧 Fixing NumPy compatibility...")
        success = fix_all_models()
        if success:
            print("\n🎉 NumPy compatibility fix completed!")
            print("💡 You can now restart your API server")
        else:
            print("\n❌ Some models could not be fixed")
            print("💡 Consider creating mock models for testing")
    
    elif choice == "2":
        print("\n🔧 Creating mock models...")
        create_mock_models()
    
    elif choice == "3":
        print("\n📋 Checking model status...")
        check_model_status()
    
    else:
        print("❌ Invalid choice")

def check_model_status():
    """Check the status of model files"""
    model_dir = Path("models/news/models")
    
    if not model_dir.exists():
        print("❌ Model directory not found")
        return
    
    print(f"📁 Model Directory: {model_dir}")
    
    # Check main models
    main_models = ["spike_model.pkl", "vectorizer.pkl", "label_encoders.pkl"]
    for model in main_models:
        model_path = model_dir / model
        if model_path.exists():
            size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {model}: {size:.1f}MB")
        else:
            print(f"❌ {model}: Not found")
    
    # Check price models
    price_models_dir = model_dir / "Price_Models"
    if price_models_dir.exists():
        price_models = list(price_models_dir.glob("price_model_*.pkl"))
        print(f"✅ Price_Models directory: {len(price_models)} models")
        for model in price_models[:5]:  # Show first 5
            size = model.stat().st_size / (1024 * 1024)  # MB
            print(f"   📄 {model.name}: {size:.1f}MB")
        if len(price_models) > 5:
            print(f"   ... and {len(price_models) - 5} more")
    else:
        print("❌ Price_Models directory: Not found")

if __name__ == "__main__":
    main() 