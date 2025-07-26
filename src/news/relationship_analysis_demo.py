#!/usr/bin/env python3
"""
Relationship Analysis Demo
Demonstrates enhanced preprocessing techniques for analyzing relationships between
news, Google search spikes, and stock price movements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from enhanced_preprocessing import EnhancedNewsPreprocessor

class RelationshipAnalyzer:
    """
    Comprehensive relationship analyzer for news, search, and stock data
    """
    
    def __init__(self):
        self.preprocessor = EnhancedNewsPreprocessor()
        self.analysis_results = {}
        
    def load_sample_data(self):
        """
        Load and prepare sample data for analysis
        """
        print("📊 Loading sample data...")
        
        # Create sample news data
        dates = pd.date_range('2014-01-01', '2019-12-31', freq='D')
        
        # News data with realistic patterns
        news_data = []
        for date in dates:
            # Create news with seasonal patterns
            is_peak_season = date.month in [3, 4, 7, 8, 12]  # Peak tourism months
            is_weekend = date.weekday() >= 5
            
            # Generate news based on patterns
            if is_peak_season:
                news_count = np.random.poisson(15)  # More news in peak season
            else:
                news_count = np.random.poisson(8)
            
            for _ in range(news_count):
                # Determine news type and sentiment
                news_type = np.random.choice(['hotel', 'tourism', 'general'], p=[0.3, 0.4, 0.3])
                sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.4, 0.2])
                
                # Create realistic news content
                if news_type == 'hotel':
                    hotels = ['John Keells', 'Aitken Spence', 'Cinnamon Hotels', 'Galadari']
                    hotel = np.random.choice(hotels)
                    heading = f"{hotel} Reports Strong Performance"
                    content = f"{hotel} has announced positive results for the quarter."
                elif news_type == 'tourism':
                    heading = "Sri Lanka Tourism Shows Growth"
                    content = "Tourism arrivals to Sri Lanka increased significantly."
                else:
                    heading = "Economic Update"
                    content = "General economic news affecting the market."
                
                news_data.append({
                    'date': date,
                    'heading': heading,
                    'content': content,
                    'source': np.random.choice(['economynext', 'dailynews', 'sundaytimes']),
                    'sentiment': sentiment,
                    'type': news_type,
                    'hotel': hotel if news_type == 'hotel' else 'Unknown'
                })
        
        news_df = pd.DataFrame(news_data)
        
        # Create search data with realistic patterns
        search_data = []
        for date in dates:
            # Base search volume
            base_volume = 100
            
            # Add seasonal effects
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Add weekly patterns
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * date.weekday() / 7)
            
            # Add random noise
            noise = np.random.normal(0, 0.1)
            
            # Calculate final volume
            volume = base_volume * seasonal_factor * weekly_factor * (1 + noise)
            
            search_data.append({
                'date': date,
                'sri_lanka_tourism': volume,
                'hotel_search': volume * 0.7,
                'travel_sri_lanka': volume * 0.8
            })
        
        search_df = pd.DataFrame(search_data)
        
        # Create stock data with realistic patterns
        stock_data = []
        base_price = 100
        
        for i, date in enumerate(dates):
            # Add trend
            trend = 0.0001 * i
            
            # Add seasonal effects
            seasonal = 0.02 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Add random walk
            if i > 0:
                random_walk = np.random.normal(0, 0.01)
            else:
                random_walk = 0
            
            # Calculate price
            price = base_price * (1 + trend + seasonal + random_walk)
            
            stock_data.append({
                'date': date,
                'Close (Rs.)': price,
                'Volume': np.random.poisson(1000)
            })
        
        stock_df = pd.DataFrame(stock_data)
        
        print(f"✅ Created sample data:")
        print(f"  News articles: {len(news_df)}")
        print(f"  Search data points: {len(search_df)}")
        print(f"  Stock data points: {len(stock_df)}")
        
        return news_df, search_df, stock_df
    
    def run_enhanced_preprocessing(self, news_df, search_df, stock_df):
        """
        Run the enhanced preprocessing pipeline
        """
        print("\n🔧 Running enhanced preprocessing...")
        
        # Run preprocessing pipeline
        processed_df = self.preprocessor.preprocess_pipeline(
            news_df, search_df, stock_df, target_col='has_spike'
        )
        
        # Analyze relationships
        self.analysis_results = self.preprocessor.analyze_relationships(processed_df)
        
        print("✅ Enhanced preprocessing completed!")
        return processed_df
    
    def analyze_temporal_patterns(self, df):
        """
        Analyze temporal patterns in the data
        """
        print("\n📅 Analyzing temporal patterns...")
        
        # Day of week patterns
        if 'day_of_week' in df.columns:
            dow_patterns = df.groupby('day_of_week')['combined_spike_score'].agg(['mean', 'std', 'count'])
            print("\nDay of Week Patterns:")
            print(dow_patterns)
        
        # Month patterns
        if 'month' in df.columns:
            month_patterns = df.groupby('month')['combined_spike_score'].agg(['mean', 'std', 'count'])
            print("\nMonth Patterns:")
            print(month_patterns)
        
        # Seasonal patterns
        if 'quarter' in df.columns:
            quarter_patterns = df.groupby('quarter')['combined_spike_score'].agg(['mean', 'std', 'count'])
            print("\nQuarter Patterns:")
            print(quarter_patterns)
    
    def analyze_cross_modal_relationships(self, df):
        """
        Analyze relationships between different data modalities
        """
        print("\n🔗 Analyzing cross-modal relationships...")
        
        # Search-stock correlations
        search_cols = [col for col in df.columns if 'search_change' in col]
        stock_cols = [col for col in df.columns if 'stock_change' in col]
        
        print("\nSearch-Stock Correlations:")
        for search_col in search_cols:
            for stock_col in stock_cols:
                if search_col != stock_col:
                    corr = df[search_col].corr(df[stock_col])
                    print(f"  {search_col} vs {stock_col}: {corr:.3f}")
        
        # Sentiment effects
        if 'sentiment_encoded' in df.columns:
            print("\nSentiment Effects:")
            sentiment_effects = df.groupby('sentiment_encoded')['combined_spike_score'].agg(['mean', 'std'])
            print(sentiment_effects)
    
    def analyze_causality_patterns(self, df):
        """
        Analyze causality patterns
        """
        print("\n🔍 Analyzing causality patterns...")
        
        # Lead-lag relationships
        lead_lag_cols = [col for col in df.columns if 'leads' in col]
        
        if lead_lag_cols:
            print("\nLead-Lag Relationships:")
            for col in lead_lag_cols:
                effect = df[col].mean()
                std_effect = df[col].std()
                positive_ratio = (df[col] > 0).mean()
                print(f"  {col}:")
                print(f"    Mean effect: {effect:.3f}")
                print(f"    Std effect: {std_effect:.3f}")
                print(f"    Positive ratio: {positive_ratio:.3f}")
    
    def create_visualizations(self, df):
        """
        Create comprehensive visualizations
        """
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Relationship Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temporal patterns
        if 'day_of_week' in df.columns:
            dow_data = df.groupby('day_of_week')['combined_spike_score'].mean()
            axes[0, 0].bar(dow_data.index, dow_data.values)
            axes[0, 0].set_title('Spike Score by Day of Week')
            axes[0, 0].set_xlabel('Day of Week')
            axes[0, 0].set_ylabel('Average Spike Score')
        
        # 2. Monthly patterns
        if 'month' in df.columns:
            month_data = df.groupby('month')['combined_spike_score'].mean()
            axes[0, 1].plot(month_data.index, month_data.values, marker='o')
            axes[0, 1].set_title('Spike Score by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average Spike Score')
        
        # 3. Search vs Stock correlation
        search_cols = [col for col in df.columns if 'search_change_7d' in col]
        stock_cols = [col for col in df.columns if 'stock_change_7d' in col]
        
        if search_cols and stock_cols:
            corr_data = []
            for search_col in search_cols:
                for stock_col in stock_cols:
                    corr = df[search_col].corr(df[stock_col])
                    corr_data.append(corr)
            
            axes[0, 2].hist(corr_data, bins=20, alpha=0.7)
            axes[0, 2].set_title('Distribution of Search-Stock Correlations')
            axes[0, 2].set_xlabel('Correlation Coefficient')
            axes[0, 2].set_ylabel('Frequency')
        
        # 4. Sentiment effects
        if 'sentiment_encoded' in df.columns:
            sentiment_data = df.groupby('sentiment_encoded')['combined_spike_score'].mean()
            axes[1, 0].bar(sentiment_data.index, sentiment_data.values)
            axes[1, 0].set_title('Spike Score by Sentiment')
            axes[1, 0].set_xlabel('Sentiment')
            axes[1, 0].set_ylabel('Average Spike Score')
        
        # 5. Feature importance
        if hasattr(self.preprocessor, 'feature_importance'):
            importance_data = pd.Series(self.preprocessor.feature_importance)
            top_features = importance_data.nlargest(10)
            axes[1, 1].barh(range(len(top_features)), top_features.values)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features.index, fontsize=8)
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance Score')
        
        # 6. Causality effects
        lead_lag_cols = [col for col in df.columns if 'leads' in col]
        if lead_lag_cols:
            causality_data = []
            for col in lead_lag_cols:
                causality_data.append(df[col].mean())
            
            axes[1, 2].bar(range(len(causality_data)), causality_data)
            axes[1, 2].set_title('Causality Effects')
            axes[1, 2].set_xlabel('Lead-Lag Relationship')
            axes[1, 2].set_ylabel('Mean Effect')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('relationship_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'relationship_analysis_visualizations.png'")
    
    def generate_insights_report(self, df):
        """
        Generate a comprehensive insights report
        """
        print("\n📋 Generating insights report...")
        
        report = []
        report.append("=" * 60)
        report.append("ENHANCED RELATIONSHIP ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"- Total samples: {len(df)}")
        report.append(f"- Total features: {len(df.columns)}")
        report.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        
        # Spike patterns
        if 'combined_spike_score' in df.columns:
            avg_spike = df['combined_spike_score'].mean()
            spike_std = df['combined_spike_score'].std()
            report.append(f"- Average spike score: {avg_spike:.3f} (±{spike_std:.3f})")
        
        # Temporal patterns
        if 'day_of_week' in df.columns:
            best_day = df.groupby('day_of_week')['combined_spike_score'].mean().idxmax()
            report.append(f"- Best day for spikes: Day {best_day}")
        
        if 'month' in df.columns:
            best_month = df.groupby('month')['combined_spike_score'].mean().idxmax()
            report.append(f"- Best month for spikes: Month {best_month}")
        
        # Correlation insights
        search_cols = [col for col in df.columns if 'search_change_7d' in col]
        stock_cols = [col for col in df.columns if 'stock_change_7d' in col]
        
        if search_cols and stock_cols:
            correlations = []
            for search_col in search_cols:
                for stock_col in stock_cols:
                    corr = df[search_col].corr(df[stock_col])
                    correlations.append(abs(corr))
            
            avg_corr = np.mean(correlations)
            report.append(f"- Average search-stock correlation: {avg_corr:.3f}")
        
        # Feature importance insights
        if hasattr(self.preprocessor, 'feature_importance'):
            top_feature = max(self.preprocessor.feature_importance.items(), key=lambda x: x[1])
            report.append(f"- Most important feature: {top_feature[0]} (score: {top_feature[1]:.3f})")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("1. Focus on temporal windows that show strongest correlations")
        report.append("2. Consider sentiment as a key predictor of market impact")
        report.append("3. Monitor search volume changes as early indicators")
        report.append("4. Use ensemble models to capture complex relationships")
        report.append("5. Implement real-time monitoring of cross-modal signals")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        with open('relationship_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        print("\n✅ Report saved as 'relationship_analysis_report.txt'")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🚀 Starting complete relationship analysis...")
        
        # 1. Load sample data
        news_df, search_df, stock_df = self.load_sample_data()
        
        # 2. Run enhanced preprocessing
        processed_df = self.run_enhanced_preprocessing(news_df, search_df, stock_df)
        
        # 3. Analyze temporal patterns
        self.analyze_temporal_patterns(processed_df)
        
        # 4. Analyze cross-modal relationships
        self.analyze_cross_modal_relationships(processed_df)
        
        # 5. Analyze causality patterns
        self.analyze_causality_patterns(processed_df)
        
        # 6. Create visualizations
        self.create_visualizations(processed_df)
        
        # 7. Generate insights report
        self.generate_insights_report(processed_df)
        
        print("\n🎉 Complete analysis finished!")
        return processed_df, self.analysis_results

def main():
    """
    Main function to run the relationship analysis demo
    """
    print("🎯 Enhanced Relationship Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RelationshipAnalyzer()
    
    # Run complete analysis
    processed_df, analysis_results = analyzer.run_complete_analysis()
    
    print("\n📊 Analysis Summary:")
    print(f"- Processed dataset shape: {processed_df.shape}")
    print(f"- Number of analysis results: {len(analysis_results)}")
    print(f"- Key insights generated: {len(analysis_results.get('correlations', {}))} correlations")
    
    return processed_df, analysis_results

if __name__ == "__main__":
    main() 