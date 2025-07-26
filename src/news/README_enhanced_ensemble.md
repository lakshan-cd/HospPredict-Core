# Enhanced Ensemble Model for Hotel Stock Price Prediction

## Overview

This enhanced ensemble system combines news sentiment analysis, Google search spike detection, and stock price movement prediction to create a comprehensive model for predicting hotel stock price movements. The system uses advanced preprocessing techniques, multi-modal feature engineering, and ensemble learning to achieve higher accuracy than traditional approaches.

## Key Features

### 🚀 Enhanced Preprocessing
- **Temporal Alignment**: Multi-window temporal alignment between news, search, and stock data
- **Advanced Spike Detection**: Multiple statistical methods for detecting meaningful spikes
- **Causality Analysis**: Lead-lag analysis and Granger causality-like tests
- **Cross-Modal Feature Engineering**: Interaction features between different data sources

### 🎯 Advanced Ensemble Architecture
- **Spike Prediction Model**: Predicts whether news will cause search volume spikes
- **Price Movement Models**: Individual models for each hotel's price direction
- **Final Ensemble Model**: Combines all predictions with advanced weighting

### 📊 Relationship Analysis
- **Correlation Analysis**: Deep analysis of relationships between modalities
- **Temporal Pattern Detection**: Seasonal and weekly pattern identification
- **Feature Importance Analysis**: Understanding which features drive predictions

## Architecture

```
News Data → Text Processing → Sentiment Analysis
     ↓
Search Data → Spike Detection → Temporal Alignment
     ↓
Stock Data → Price Analysis → Volatility Calculation
     ↓
Enhanced Preprocessing → Feature Engineering → Causality Analysis
     ↓
Ensemble Models → Spike Model + Price Models → Final Ensemble
     ↓
Prediction Output → Confidence Scores → Hotel-Specific Predictions
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_enhanced.txt
```

2. Set up the project structure:
```
src/news/
├── enhanced_ensemble_model.py      # Main ensemble model
├── enhanced_preprocessing.py       # Advanced preprocessing
├── train_enhanced_ensemble.py      # Training script
├── predict_with_ensemble.py        # Prediction script
├── relationship_analysis_demo.py   # Analysis demo
└── requirements_enhanced.txt       # Dependencies
```

## Usage

### 1. Training the Enhanced Ensemble Model

```python
from src.news.train_enhanced_ensemble import main

# Run the training pipeline
main()
```

The training script will:
- Load and preprocess your data
- Create advanced features
- Train individual models
- Build the ensemble
- Save all models

### 2. Making Predictions

```python
from src.news.predict_with_ensemble import NewsPredictor

# Initialize predictor
predictor = NewsPredictor()

# Make prediction for a single article
prediction = predictor.predict_single_article(
    heading="John Keells Hotels Reports Strong Q3 Performance",
    content="John Keells Hotels PLC announced impressive third-quarter results...",
    source="economynext"
)

print(f"Spike Probability: {prediction['spike_probability']:.3f}")
print(f"Ensemble Probability: {prediction['ensemble_probability']:.3f}")
```

### 3. Running Relationship Analysis

```python
from src.news.relationship_analysis_demo import RelationshipAnalyzer

# Initialize analyzer
analyzer = RelationshipAnalyzer()

# Run complete analysis
processed_df, analysis_results = analyzer.run_complete_analysis()
```

## Enhanced Preprocessing Techniques

### 1. Temporal Alignment
- **Multi-window Analysis**: Analyzes relationships across different time windows (1, 3, 7, 14 days)
- **Lag Analysis**: Examines lead-lag relationships between news, search, and stock movements
- **Seasonal Decomposition**: Accounts for seasonal patterns in tourism and stock markets

### 2. Advanced Spike Detection
- **Statistical Methods**: Z-score based spike detection with rolling statistics
- **Volatility-based Detection**: Identifies spikes based on volatility changes
- **Multi-timeframe Analysis**: Combines signals from different time windows
- **Change Point Detection**: Identifies structural breaks in the data

### 3. Causality Analysis
- **Granger-like Causality**: Tests whether one variable helps predict another
- **Lead-Lag Analysis**: Examines temporal precedence between variables
- **Cross-correlation Analysis**: Measures correlation at different time lags
- **Information Flow Analysis**: Uses mutual information to measure relationships

### 4. Feature Engineering
- **Interaction Features**: Creates features that capture relationships between modalities
- **Ratio Features**: Calculates ratios between different variables
- **Polynomial Features**: Captures non-linear relationships
- **Rolling Features**: Creates moving average and volatility features
- **Lag Features**: Incorporates historical information

## Model Architecture

### Spike Prediction Model
- **Ensemble of Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Network
- **Soft Voting**: Combines predictions using probability scores
- **SMOTE Balancing**: Handles class imbalance in spike detection

### Price Movement Models
- **Hotel-Specific Models**: Individual models for each hotel
- **Multi-class Classification**: Predicts up/down/neutral movements
- **Dynamic Thresholds**: Adjusts thresholds based on hotel volatility

### Final Ensemble Model
- **Meta-learner**: Combines spike and price predictions
- **Feature Integration**: Incorporates original features with predictions
- **Confidence Scoring**: Provides confidence levels for predictions

## Key Improvements Over Original Approach

### 1. Enhanced Feature Engineering
- **Cross-modal Interactions**: Captures relationships between news, search, and stock data
- **Temporal Features**: Accounts for time-based patterns
- **Statistical Features**: Incorporates rolling statistics and volatility measures

### 2. Advanced Relationship Detection
- **Causality Analysis**: Goes beyond correlation to understand causal relationships
- **Multi-window Analysis**: Examines relationships across different time horizons
- **Lead-lag Analysis**: Identifies which variable leads which

### 3. Improved Model Architecture
- **Ensemble Methods**: Combines multiple models for better performance
- **Hotel-Specific Models**: Tailored models for each hotel's characteristics
- **Meta-learning**: Final ensemble that learns from individual model predictions

### 4. Better Data Alignment
- **Temporal Alignment**: Properly aligns data from different sources
- **Missing Data Handling**: Robust handling of missing values
- **Outlier Detection**: Identifies and handles outliers appropriately

## Configuration

The system can be configured through the `config` parameter:

```python
config = {
    'temporal_windows': [1, 3, 7, 14],  # Days before/after news
    'spike_threshold': 2.0,  # Standard deviations for spike detection
    'correlation_threshold': 0.3,  # Minimum correlation to consider relationship
    'min_samples': 10,  # Minimum samples for statistical tests
    'rolling_windows': [3, 7, 14, 30],  # Rolling window sizes
    'feature_selection_k': 50,  # Top k features to select
    'causality_lag_max': 7,  # Maximum lag for causality analysis
}

preprocessor = EnhancedNewsPreprocessor(config)
```

## Performance Metrics

The system provides comprehensive performance metrics:

- **Spike Model**: Precision, Recall, F1-Score, AUC
- **Price Models**: Per-hotel performance metrics
- **Ensemble Model**: Overall prediction accuracy
- **Feature Importance**: Understanding of which features drive predictions

## Visualization and Analysis

The system includes comprehensive visualization capabilities:

- **Temporal Patterns**: Day-of-week and monthly patterns
- **Correlation Analysis**: Search-stock correlation distributions
- **Feature Importance**: Top features ranked by importance
- **Causality Effects**: Lead-lag relationship visualizations

## Best Practices

### 1. Data Quality
- Ensure consistent date formats across all data sources
- Handle missing values appropriately
- Validate data ranges and remove outliers

### 2. Feature Engineering
- Start with basic features and gradually add complexity
- Monitor feature importance to avoid overfitting
- Use cross-validation to validate feature selection

### 3. Model Training
- Use stratified sampling for imbalanced datasets
- Implement early stopping to prevent overfitting
- Monitor validation metrics during training

### 4. Evaluation
- Use multiple metrics for comprehensive evaluation
- Perform temporal validation (train on past, test on future)
- Analyze model interpretability and feature importance

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce feature selection or use smaller temporal windows
2. **Overfitting**: Increase regularization or reduce model complexity
3. **Poor Performance**: Check data quality and feature engineering
4. **Slow Training**: Use parallel processing or reduce ensemble size

### Performance Optimization

1. **Feature Selection**: Use the built-in feature selection to reduce dimensionality
2. **Parallel Processing**: Enable parallel processing for faster training
3. **Data Sampling**: Use stratified sampling for large datasets
4. **Model Caching**: Save and load models to avoid retraining

## Future Enhancements

1. **Deep Learning Integration**: Add transformer models for text processing
2. **Real-time Processing**: Implement streaming data processing
3. **Advanced Causality**: Integrate more sophisticated causality methods
4. **Multi-modal Fusion**: Better integration of different data modalities

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the repository or contact the development team. 