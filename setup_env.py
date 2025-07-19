#!/usr/bin/env python3
"""
Setup script to help users create their .env file from .env.example
"""

import os
import shutil

def setup_environment():
    """Set up environment variables file."""
    
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    if not os.path.exists('.env.example'):
        print("❌ .env.example file not found!")
        print("Please create .env.example first.")
        return
    
    try:
        # Copy .env.example to .env
        shutil.copy('.env.example', '.env')
        print("✅ .env file created successfully!")
        print("\n📝 Next steps:")
        print("1. Edit .env file with your actual Neo4j credentials")
        print("2. Update other configuration values as needed")
        print("3. Run 'python main.py' to start the API")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    setup_environment() 