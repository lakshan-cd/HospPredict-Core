#!/usr/bin/env python3
"""
Test script for News Prediction API
Tests the integrated news prediction endpoints
"""

import requests
import json
import time
from datetime import datetime

class NewsAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test API health"""
        print("🔍 Testing API Health...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"✅ Health Check: {response.json()}")
            return True
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return False
    
    def test_root(self):
        """Test root endpoint"""
        print("\n🏠 Testing Root Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            data = response.json()
            print(f"✅ Root Endpoint: {data['message']}")
            print(f"📋 Features: {len(data['features'])} features available")
            return True
        except Exception as e:
            print(f"❌ Root Endpoint Failed: {e}")
            return False
    
    def test_news_health(self):
        """Test news prediction health"""
        print("\n📰 Testing News Prediction Health...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/news/health")
            data = response.json()
            print(f"✅ News Health: {data}")
            return True
        except Exception as e:
            print(f"❌ News Health Failed: {e}")
            return False
    
    def test_single_prediction(self):
        """Test single prediction"""
        print("\n🎯 Testing Single Prediction...")
        try:
            data = {
                "date": "2024-01-15",
                "heading": "Tourism surge expected after visa reforms",
                "text": "Sri Lanka tourism is expected to see a massive surge due to easier visa rules. The new policy will attract more international visitors and boost hotel bookings across the country.",
                "source": "economynext"
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/news/predict", json=data)
            result = response.json()
            
            print(f"✅ Single Prediction: {result['spike_prediction']} (confidence: {result['spike_confidence']:.3f})")
            print(f"🏨 Hotel Predictions: {len(result['hotel_predictions'])} hotels")
            
            # Show first few hotel predictions
            for i, (hotel, pred) in enumerate(list(result['hotel_predictions'].items())[:3]):
                print(f"   {hotel}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
            
            return True
        except Exception as e:
            print(f"❌ Single Prediction Failed: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        print("\n📦 Testing Batch Prediction...")
        try:
            articles = [
                {
                    "date": "2024-01-15",
                    "heading": "Tourism surge expected after visa reforms",
                    "text": "Sri Lanka tourism is expected to see a massive surge due to easier visa rules.",
                    "source": "economynext"
                },
                {
                    "date": "2024-01-15",
                    "heading": "Hotel profits decline due to economic challenges",
                    "text": "Major hotels report declining profits due to economic challenges and reduced tourist arrivals.",
                    "source": "dailynews"
                }
            ]
            
            data = {"articles": articles}
            response = self.session.post(f"{self.base_url}/api/v1/news/predict/batch", json=data)
            result = response.json()
            
            print(f"✅ Batch Prediction: {result['summary']['successful_predictions']}/{result['summary']['total_articles']} successful")
            print(f"📊 Summary: {result['summary']}")
            
            return True
        except Exception as e:
            print(f"❌ Batch Prediction Failed: {e}")
            return False
    
    def test_hotels_list(self):
        """Test hotels list endpoint"""
        print("\n🏨 Testing Hotels List...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/news/hotels")
            data = response.json()
            
            print(f"✅ Hotels List: {data['count']} hotels available")
            print(f"📋 First 5 hotels: {data['hotels'][:5]}")
            
            return True
        except Exception as e:
            print(f"❌ Hotels List Failed: {e}")
            return False
    
    def test_features_info(self):
        """Test features info endpoint"""
        print("\n🔧 Testing Features Info...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/news/features")
            data = response.json()
            
            print(f"✅ Features Info: {len(data)} feature categories")
            for category, features in data.items():
                print(f"   {category}: {len(features)} features")
            
            return True
        except Exception as e:
            print(f"❌ Features Info Failed: {e}")
            return False
    
    def test_analytics(self):
        """Test analytics endpoints"""
        print("\n📊 Testing Analytics...")
        try:
            # Test performance metrics
            response = self.session.get(f"{self.base_url}/api/v1/news/analytics/performance")
            performance = response.json()
            print(f"✅ Performance Metrics: {len(performance)} models")
            
            # Test trends
            response = self.session.get(f"{self.base_url}/api/v1/news/analytics/trends?days=7")
            trends = response.json()
            print(f"✅ Trends: {len(trends)} data points")
            
            # Test hotel insights
            response = self.session.get(f"{self.base_url}/api/v1/news/analytics/hotels/insights")
            insights = response.json()
            print(f"✅ Hotel Insights: {len(insights)} hotels")
            
            return True
        except Exception as e:
            print(f"❌ Analytics Failed: {e}")
            return False
    
    def test_admin(self):
        """Test admin endpoints"""
        print("\n⚙️ Testing Admin...")
        try:
            # Test system info
            response = self.session.get(f"{self.base_url}/api/v1/news/admin/system/info")
            system_info = response.json()
            print(f"✅ System Info: {system_info['system_status']}")
            
            # Test models status
            response = self.session.get(f"{self.base_url}/api/v1/news/admin/models/status")
            models_status = response.json()
            print(f"✅ Models Status: {len(models_status)} models")
            
            return True
        except Exception as e:
            print(f"❌ Admin Failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting News Prediction API Tests...")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health),
            ("Root Endpoint", self.test_root),
            ("News Health", self.test_news_health),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Hotels List", self.test_hotels_list),
            ("Features Info", self.test_features_info),
            ("Analytics", self.test_analytics),
            ("Admin", self.test_admin),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} Exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! News Prediction API is working correctly.")
        else:
            print("⚠️ Some tests failed. Please check the API configuration.")
        
        return passed == total

def main():
    """Main test function"""
    tester = NewsAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🔗 API Endpoints Available:")
        print("   • Single Prediction: POST /api/v1/news/predict")
        print("   • Batch Prediction: POST /api/v1/news/predict/batch")
        print("   • File Upload: POST /api/v1/news/predict/upload")
        print("   • Analytics: GET /api/v1/news/analytics/*")
        print("   • Admin: GET /api/v1/news/admin/*")
        print("   • WebSocket: WebSocket /api/v1/news/ws")
        print("\n📚 Documentation: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Please check:")
        print("   1. API server is running on http://localhost:8000")
        print("   2. Models are loaded correctly")
        print("   3. All dependencies are installed")

if __name__ == "__main__":
    main() 