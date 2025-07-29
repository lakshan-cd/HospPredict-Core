#!/usr/bin/env python3
"""
Script to install VADER sentiment analysis for the news prediction API
"""

import subprocess
import sys

def install_vader():
    """Install VADER sentiment analysis"""
    try:
        print("🔧 Installing VADER sentiment analysis...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
        print("✅ VADER sentiment analysis installed successfully!")
        
        # Test import
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            test_score = analyzer.polarity_scores("This is a positive test.")
            print(f"✅ VADER test successful: {test_score}")
            return True
        except Exception as e:
            print(f"❌ VADER import test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to install VADER: {e}")
        return False

def main():
    """Main function"""
    print("🚀 VADER Sentiment Analysis Installer")
    print("=" * 50)
    
    success = install_vader()
    
    if success:
        print("\n🎉 VADER installed successfully!")
        print("💡 You can now restart your API server")
        print("💡 The API will use VADER sentiment analysis (same as training)")
    else:
        print("\n❌ VADER installation failed")
        print("💡 The API will fall back to simple keyword-based sentiment analysis")
        print("💡 For best results, install VADER manually: pip install vaderSentiment")

if __name__ == "__main__":
    main() 