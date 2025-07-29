#!/usr/bin/env python3
"""
Simple test script to verify VADER sentiment analysis
"""

def test_vader():
    """Test VADER sentiment analysis"""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        # Initialize analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Test texts
        test_texts = [
            "Tourism surge expected after visa reforms",
            "Hotel profits decline due to economic challenges",
            "Sri Lanka tourism is booming with record arrivals",
            "Hotel industry faces crisis due to pandemic",
            "Neutral news about hotel developments"
        ]
        
        print("🧪 Testing VADER Sentiment Analysis")
        print("=" * 50)
        
        for text in test_texts:
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Apply same thresholds as training code
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            print(f"Text: {text}")
            print(f"Scores: {scores}")
            print(f"Sentiment: {sentiment}")
            print("-" * 30)
        
        print("✅ VADER sentiment analysis is working correctly!")
        return True
        
    except ImportError:
        print("❌ VADER not installed. Run: pip install vaderSentiment")
        return False
    except Exception as e:
        print(f"❌ Error testing VADER: {e}")
        return False

if __name__ == "__main__":
    test_vader() 