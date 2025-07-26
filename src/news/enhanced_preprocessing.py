import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNewsPreprocessor:
    """
    Enhanced preprocessing for news, Google search spikes, and stock price relationships
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for preprocessing"""
        return {
            'temporal_windows': [1, 3, 7, 14],  # Days before/after news
            'spike_threshold': 2.0,  # Standard deviations for spike detection
            'correlation_threshold': 0.3,  # Minimum correlation to consider relationship
            'min_samples': 10,  # Minimum samples for statistical tests
            'rolling_windows': [3, 7, 14, 30],  # Rolling window sizes
            'feature_selection_k': 50,  # Top k features to select
            'causality_lag_max': 7,  # Maximum lag for causality analysis
        }
    
    def temporal_alignment(self, news_df: pd.DataFrame, search_df: pd.DataFrame, 
                          stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced temporal alignment with multiple time windows and lag analysis
        """
        logger.info("🕐 Performing advanced temporal alignment...")
        
        # Ensure datetime columns
        news_df['date'] = pd.to_datetime(news_df['date'])
        search_df['date'] = pd.to_datetime(search_df['date'])
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        # Create temporal features for news
        news_df = self._add_temporal_features(news_df)
        
        # Align data with multiple time windows
        aligned_data = []
        
        for window in self.config['temporal_windows']:
            logger.info(f"Processing {window}-day window...")
            
            # Create lagged and lead features
            window_data = self._create_window_features(
                news_df, search_df, stock_df, window
            )
            
            # Add window identifier
            window_data['temporal_window'] = window
            aligned_data.append(window_data)
        
        # Combine all windows
        combined_df = pd.concat(aligned_data, ignore_index=True)
        
        logger.info(f"✅ Temporal alignment completed. Shape: {combined_df.shape}")
        return combined_df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive temporal features"""
        df = df.copy()
        
        # Basic temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Advanced temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_holiday_season'] = df['month'].isin([12, 1, 2]).astype(int)  # Dec-Feb
        df['is_peak_season'] = df['month'].isin([3, 4, 7, 8, 12]).astype(int)  # Peak tourism months
        
        # Cyclical encoding for periodic features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_window_features(self, news_df: pd.DataFrame, search_df: pd.DataFrame, 
                               stock_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Create features for a specific time window"""
        
        window_features = []
        
        for idx, news_row in news_df.iterrows():
            news_date = news_row['date']
            
            # Define time windows
            pre_window_start = news_date - timedelta(days=window)
            pre_window_end = news_date - timedelta(days=1)
            post_window_start = news_date + timedelta(days=1)
            post_window_end = news_date + timedelta(days=window)
            
            # Extract search data for this window
            pre_search = search_df[
                (search_df['date'] >= pre_window_start) & 
                (search_df['date'] <= pre_window_end)
            ]
            post_search = search_df[
                (search_df['date'] >= post_window_start) & 
                (search_df['date'] <= post_window_end)
            ]
            
            # Extract stock data for this window
            pre_stock = stock_df[
                (stock_df['date'] >= pre_window_start) & 
                (stock_df['date'] <= pre_window_end)
            ]
            post_stock = stock_df[
                (stock_df['date'] >= post_window_start) & 
                (stock_df['date'] <= post_window_end)
            ]
            
            # Create window-specific features
            window_row = self._extract_window_features(
                news_row, pre_search, post_search, pre_stock, post_stock, window
            )
            window_features.append(window_row)
        
        return pd.DataFrame(window_features)
    
    def _extract_window_features(self, news_row: pd.Series, pre_search: pd.DataFrame, 
                                post_search: pd.DataFrame, pre_stock: pd.DataFrame, 
                                post_stock: pd.DataFrame, window: int) -> Dict:
        """Extract comprehensive features for a time window"""
        
        features = news_row.to_dict()
        features['temporal_window'] = window
        
        # Search volume features
        features.update(self._extract_search_features(pre_search, post_search, window))
        
        # Stock price features
        features.update(self._extract_stock_features(pre_stock, post_stock, window))
        
        # Cross-modal features
        features.update(self._extract_cross_modal_features(
            pre_search, post_search, pre_stock, post_stock, window
        ))
        
        return features
    
    def _extract_search_features(self, pre_search: pd.DataFrame, 
                                post_search: pd.DataFrame, window: int) -> Dict:
        """Extract search volume related features"""
        features = {}
        
        # Pre-news search features
        if not pre_search.empty:
            features[f'pre_search_mean_{window}d'] = pre_search.select_dtypes(include=[np.number]).mean().mean()
            features[f'pre_search_std_{window}d'] = pre_search.select_dtypes(include=[np.number]).std().mean()
            features[f'pre_search_trend_{window}d'] = self._calculate_trend(pre_search.select_dtypes(include=[np.number]))
            features[f'pre_search_volatility_{window}d'] = pre_search.select_dtypes(include=[np.number]).std().mean() / (pre_search.select_dtypes(include=[np.number]).mean().mean() + 1e-8)
        else:
            features[f'pre_search_mean_{window}d'] = 0
            features[f'pre_search_std_{window}d'] = 0
            features[f'pre_search_trend_{window}d'] = 0
            features[f'pre_search_volatility_{window}d'] = 0
        
        # Post-news search features
        if not post_search.empty:
            features[f'post_search_mean_{window}d'] = post_search.select_dtypes(include=[np.number]).mean().mean()
            features[f'post_search_std_{window}d'] = post_search.select_dtypes(include=[np.number]).std().mean()
            features[f'post_search_trend_{window}d'] = self._calculate_trend(post_search.select_dtypes(include=[np.number]))
            features[f'post_search_volatility_{window}d'] = post_search.select_dtypes(include=[np.number]).std().mean() / (post_search.select_dtypes(include=[np.number]).mean().mean() + 1e-8)
        else:
            features[f'post_search_mean_{window}d'] = 0
            features[f'post_search_std_{window}d'] = 0
            features[f'post_search_trend_{window}d'] = 0
            features[f'post_search_volatility_{window}d'] = 0
        
        # Search change features
        features[f'search_change_{window}d'] = features[f'post_search_mean_{window}d'] - features[f'pre_search_mean_{window}d']
        features[f'search_change_pct_{window}d'] = (features[f'search_change_{window}d'] / (features[f'pre_search_mean_{window}d'] + 1e-8)) * 100
        
        return features
    
    def _extract_stock_features(self, pre_stock: pd.DataFrame, 
                               post_stock: pd.DataFrame, window: int) -> Dict:
        """Extract stock price related features"""
        features = {}
        
        # Pre-news stock features
        if not pre_stock.empty:
            features[f'pre_stock_mean_{window}d'] = pre_stock['Close (Rs.)'].mean()
            features[f'pre_stock_std_{window}d'] = pre_stock['Close (Rs.)'].std()
            features[f'pre_stock_trend_{window}d'] = self._calculate_trend(pre_stock[['Close (Rs.)']])
            features[f'pre_stock_volatility_{window}d'] = pre_stock['Close (Rs.)'].std() / (pre_stock['Close (Rs.)'].mean() + 1e-8)
        else:
            features[f'pre_stock_mean_{window}d'] = 0
            features[f'pre_stock_std_{window}d'] = 0
            features[f'pre_stock_trend_{window}d'] = 0
            features[f'pre_stock_volatility_{window}d'] = 0
        
        # Post-news stock features
        if not post_stock.empty:
            features[f'post_stock_mean_{window}d'] = post_stock['Close (Rs.)'].mean()
            features[f'post_stock_std_{window}d'] = post_stock['Close (Rs.)'].std()
            features[f'post_stock_trend_{window}d'] = self._calculate_trend(post_stock[['Close (Rs.)']])
            features[f'post_stock_volatility_{window}d'] = post_stock['Close (Rs.)'].std() / (post_stock['Close (Rs.)'].mean() + 1e-8)
        else:
            features[f'post_stock_mean_{window}d'] = 0
            features[f'post_stock_std_{window}d'] = 0
            features[f'post_stock_trend_{window}d'] = 0
            features[f'post_stock_volatility_{window}d'] = 0
        
        # Stock change features
        features[f'stock_change_{window}d'] = features[f'post_stock_mean_{window}d'] - features[f'pre_stock_mean_{window}d']
        features[f'stock_change_pct_{window}d'] = (features[f'stock_change_{window}d'] / (features[f'pre_stock_mean_{window}d'] + 1e-8)) * 100
        
        return features
    
    def _extract_cross_modal_features(self, pre_search: pd.DataFrame, post_search: pd.DataFrame,
                                    pre_stock: pd.DataFrame, post_stock: pd.DataFrame, 
                                    window: int) -> Dict:
        """Extract cross-modal relationship features"""
        features = {}
        
        # Search-stock correlation features
        if not pre_search.empty and not pre_stock.empty:
            # Align dates for correlation calculation
            aligned_data = self._align_for_correlation(pre_search, pre_stock)
            if len(aligned_data) > 1:
                search_cols = [col for col in aligned_data.columns if col not in ['date', 'Close (Rs.)']]
                correlations = []
                for col in search_cols:
                    corr, _ = pearsonr(aligned_data[col], aligned_data['Close (Rs.)'])
                    correlations.append(corr)
                features[f'pre_search_stock_corr_{window}d'] = np.mean(correlations)
            else:
                features[f'pre_search_stock_corr_{window}d'] = 0
        else:
            features[f'pre_search_stock_corr_{window}d'] = 0
        
        # Post-news correlation
        if not post_search.empty and not post_stock.empty:
            aligned_data = self._align_for_correlation(post_search, post_stock)
            if len(aligned_data) > 1:
                search_cols = [col for col in aligned_data.columns if col not in ['date', 'Close (Rs.)']]
                correlations = []
                for col in search_cols:
                    corr, _ = pearsonr(aligned_data[col], aligned_data['Close (Rs.)'])
                    correlations.append(corr)
                features[f'post_search_stock_corr_{window}d'] = np.mean(correlations)
            else:
                features[f'post_search_stock_corr_{window}d'] = 0
        else:
            features[f'post_search_stock_corr_{window}d'] = 0
        
        # Correlation change
        features[f'search_stock_corr_change_{window}d'] = (
            features[f'post_search_stock_corr_{window}d'] - features[f'pre_search_stock_corr_{window}d']
        )
        
        return features
    
    def _align_for_correlation(self, search_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        """Align search and stock data for correlation calculation"""
        if search_df.empty or stock_df.empty:
            return pd.DataFrame()
        
        # Merge on date
        merged = pd.merge(search_df, stock_df[['date', 'Close (Rs.)']], on='date', how='inner')
        return merged.dropna()
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend using linear regression slope"""
        if df.empty or len(df) < 2:
            return 0
        
        x = np.arange(len(df))
        y = df.values.flatten()
        
        if len(y) < 2:
            return 0
        
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def advanced_spike_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced spike detection using multiple methods
        """
        logger.info("📈 Performing advanced spike detection...")
        
        df = df.copy()
        
        # 1. Statistical spike detection (Z-score based)
        df = self._statistical_spike_detection(df)
        
        # 2. Volatility-based spike detection
        df = self._volatility_spike_detection(df)
        
        # 3. Change point detection
        df = self._change_point_detection(df)
        
        # 4. Multi-timeframe spike detection
        df = self._multi_timeframe_spike_detection(df)
        
        # 5. Combine spike signals
        df = self._combine_spike_signals(df)
        
        logger.info("✅ Advanced spike detection completed")
        return df
    
    def _statistical_spike_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical spike detection using Z-scores"""
        
        # Get numeric columns (excluding date and categorical)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith(('pre_', 'post_', 'search_', 'stock_')):
                # Calculate rolling statistics
                rolling_mean = df[col].rolling(window=30, min_periods=1).mean()
                rolling_std = df[col].rolling(window=30, min_periods=1).std()
                
                # Calculate Z-score
                z_score = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                
                # Detect spikes
                df[f'{col}_z_spike'] = (z_score > self.config['spike_threshold']).astype(int)
                df[f'{col}_z_score'] = z_score
        
        return df
    
    def _volatility_spike_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based spike detection"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith(('pre_', 'post_', 'search_', 'stock_')):
                # Calculate rolling volatility
                rolling_vol = df[col].rolling(window=14, min_periods=1).std()
                vol_mean = rolling_vol.rolling(window=30, min_periods=1).mean()
                vol_std = rolling_vol.rolling(window=30, min_periods=1).std()
                
                # Detect volatility spikes
                vol_z_score = (rolling_vol - vol_mean) / (vol_std + 1e-8)
                df[f'{col}_vol_spike'] = (vol_z_score > self.config['spike_threshold']).astype(int)
        
        return df
    
    def _change_point_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple change point detection using rolling statistics"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith(('pre_', 'post_', 'search_', 'stock_')):
                # Calculate rolling mean with different windows
                mean_short = df[col].rolling(window=7, min_periods=1).mean()
                mean_long = df[col].rolling(window=30, min_periods=1).mean()
                
                # Detect change points
                change_ratio = abs(mean_short - mean_long) / (mean_long + 1e-8)
                df[f'{col}_change_point'] = (change_ratio > 0.1).astype(int)
        
        return df
    
    def _multi_timeframe_spike_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe spike detection"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith(('pre_', 'post_', 'search_', 'stock_')):
                spike_signals = []
                
                for window in [7, 14, 30]:
                    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = df[col].rolling(window=window, min_periods=1).std()
                    z_score = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                    spike_signals.append(z_score > self.config['spike_threshold'])
                
                # Combine signals (majority vote)
                combined_signal = np.mean(spike_signals, axis=0) > 0.5
                df[f'{col}_multi_spike'] = combined_signal.astype(int)
        
        return df
    
    def _combine_spike_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine different spike detection signals"""
        
        # Get all spike columns
        spike_cols = [col for col in df.columns if col.endswith(('_spike', '_change_point'))]
        
        # Create combined spike signal
        if spike_cols:
            spike_matrix = df[spike_cols].fillna(0)
            df['combined_spike_score'] = spike_matrix.sum(axis=1)
            df['has_spike'] = (df['combined_spike_score'] > 0).astype(int)
        
        return df
    
    def causality_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform causality analysis between news, search, and stock movements
        """
        logger.info("🔍 Performing causality analysis...")
        
        df = df.copy()
        
        # 1. Granger causality-like analysis
        df = self._granger_like_analysis(df)
        
        # 2. Lead-lag analysis
        df = self._lead_lag_analysis(df)
        
        # 3. Cross-correlation analysis
        df = self._cross_correlation_analysis(df)
        
        # 4. Information flow analysis
        df = self._information_flow_analysis(df)
        
        logger.info("✅ Causality analysis completed")
        return df
    
    def _granger_like_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified Granger causality analysis"""
        
        # For each temporal window, analyze lead-lag relationships
        for window in self.config['temporal_windows']:
            # Search leads stock
            search_col = f'search_change_{window}d'
            stock_col = f'stock_change_{window}d'
            
            if search_col in df.columns and stock_col in df.columns:
                # Calculate correlation between search change and stock change
                corr, p_value = pearsonr(df[search_col].fillna(0), df[stock_col].fillna(0))
                df[f'search_leads_stock_{window}d'] = corr if p_value < 0.05 else 0
                
                # Stock leads search
                df[f'stock_leads_search_{window}d'] = corr if p_value < 0.05 else 0
        
        return df
    
    def _lead_lag_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lead-lag analysis with multiple lags"""
        
        for lag in range(1, self.config['causality_lag_max'] + 1):
            for window in self.config['temporal_windows']:
                search_col = f'search_change_{window}d'
                stock_col = f'stock_change_{window}d'
                
                if search_col in df.columns and stock_col in df.columns:
                    # Lagged correlation
                    if len(df) > lag:
                        corr, _ = pearsonr(
                            df[search_col].fillna(0).iloc[:-lag], 
                            df[stock_col].fillna(0).iloc[lag:]
                        )
                        df[f'search_lag{lag}_stock_{window}d'] = corr
        
        return df
    
    def _cross_correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-correlation analysis"""
        
        for window in self.config['temporal_windows']:
            search_col = f'search_change_{window}d'
            stock_col = f'stock_change_{window}d'
            
            if search_col in df.columns and stock_col in df.columns:
                # Calculate cross-correlation
                x = df[search_col].fillna(0).values
                y = df[stock_col].fillna(0).values
                
                if len(x) > 10 and len(y) > 10:
                    # Simple cross-correlation
                    corr_matrix = np.corrcoef(x, y)
                    if corr_matrix.shape == (2, 2):
                        df[f'cross_corr_{window}d'] = corr_matrix[0, 1]
        
        return df
    
    def _information_flow_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Information flow analysis using mutual information"""
        
        for window in self.config['temporal_windows']:
            search_col = f'search_change_{window}d'
            stock_col = f'stock_change_{window}d'
            
            if search_col in df.columns and stock_col in df.columns:
                # Calculate mutual information
                X = df[search_col].fillna(0).values.reshape(-1, 1)
                y = df[stock_col].fillna(0).values
                
                if len(X) > 10 and len(y) > 10:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                    df[f'mutual_info_{window}d'] = mi_scores[0]
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for relationship detection
        """
        logger.info("🔧 Performing advanced feature engineering...")
        
        df = df.copy()
        
        # 1. Interaction features
        df = self._create_interaction_features(df)
        
        # 2. Ratio features
        df = self._create_ratio_features(df)
        
        # 3. Polynomial features
        df = self._create_polynomial_features(df)
        
        # 4. Rolling features
        df = self._create_rolling_features(df)
        
        # 5. Lag features
        df = self._create_lag_features(df)
        
        # 6. Statistical features
        df = self._create_statistical_features(df)
        
        logger.info("✅ Feature engineering completed")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different modalities"""
        
        # Search-stock interactions
        search_cols = [col for col in df.columns if 'search_change' in col]
        stock_cols = [col for col in df.columns if 'stock_change' in col]
        
        for search_col in search_cols:
            for stock_col in stock_cols:
                if search_col != stock_col:
                    interaction_name = f'interaction_{search_col}_{stock_col}'
                    df[interaction_name] = df[search_col].fillna(0) * df[stock_col].fillna(0)
        
        # Sentiment interactions
        if 'sentiment_encoded' in df.columns:
            for col in search_cols + stock_cols:
                df[f'sentiment_{col}_interaction'] = df['sentiment_encoded'] * df[col].fillna(0)
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features"""
        
        # Search to stock ratios
        search_cols = [col for col in df.columns if 'search_change' in col]
        stock_cols = [col for col in df.columns if 'stock_change' in col]
        
        for search_col in search_cols:
            for stock_col in stock_cols:
                if search_col != stock_col:
                    ratio_name = f'ratio_{search_col}_{stock_col}'
                    df[ratio_name] = df[search_col].fillna(0) / (df[stock_col].fillna(0) + 1e-8)
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for important variables"""
        
        important_cols = [
            'sentiment_encoded', 'search_change_7d', 'stock_change_7d',
            'combined_spike_score', 'mutual_info_7d'
        ]
        
        for col in important_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col].fillna(0) ** 2
                df[f'{col}_cubed'] = df[col].fillna(0) ** 3
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        
        for window in self.config['rolling_windows']:
            for col in df.columns:
                if any(keyword in col for keyword in ['change', 'spike', 'corr']):
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        
        for lag in [1, 2, 3]:
            for col in df.columns:
                if any(keyword in col for keyword in ['change', 'spike', 'corr']):
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        
        # Percentile features
        for col in df.columns:
            if any(keyword in col for keyword in ['change', 'spike', 'corr']):
                df[f'{col}_percentile'] = df[col].rank(pct=True)
        
        # Z-score normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def feature_selection(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Advanced feature selection
        """
        logger.info("🎯 Performing feature selection...")
        
        # Prepare data
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # 1. Correlation-based selection
        correlations = []
        for col in feature_cols:
            corr, _ = pearsonr(X[col], y)
            correlations.append(abs(corr))
        
        # 2. Mutual information selection
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 3. Statistical tests
        f_scores, _ = f_regression(X, y)
        
        # Combine scores
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'correlation': correlations,
            'mutual_info': mi_scores,
            'f_score': f_scores
        })
        
        # Normalize scores
        feature_scores['correlation_norm'] = feature_scores['correlation'] / feature_scores['correlation'].max()
        feature_scores['mutual_info_norm'] = feature_scores['mutual_info'] / feature_scores['mutual_info'].max()
        feature_scores['f_score_norm'] = feature_scores['f_score'] / feature_scores['f_score'].max()
        
        # Combined score
        feature_scores['combined_score'] = (
            feature_scores['correlation_norm'] + 
            feature_scores['mutual_info_norm'] + 
            feature_scores['f_score_norm']
        ) / 3
        
        # Select top features
        top_features = feature_scores.nlargest(self.config['feature_selection_k'], 'combined_score')['feature'].tolist()
        
        # Store feature importance
        self.feature_importance = feature_scores.set_index('feature')['combined_score'].to_dict()
        
        # Return selected features
        selected_cols = [target_col] + top_features
        selected_df = df[selected_cols].copy()
        
        logger.info(f"✅ Selected {len(top_features)} features out of {len(feature_cols)}")
        return selected_df
    
    def preprocess_pipeline(self, news_df: pd.DataFrame, search_df: pd.DataFrame, 
                           stock_df: pd.DataFrame, target_col: str = 'spike_linked') -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        """
        logger.info("🚀 Starting complete preprocessing pipeline...")
        
        # 1. Temporal alignment
        df = self.temporal_alignment(news_df, search_df, stock_df)
        
        # 2. Advanced spike detection
        df = self.advanced_spike_detection(df)
        
        # 3. Causality analysis
        df = self.causality_analysis(df)
        
        # 4. Feature engineering
        df = self.feature_engineering(df)
        
        # 5. Feature selection
        if target_col in df.columns:
            df = self.feature_selection(df, target_col)
        
        # 6. Final cleaning
        df = df.fillna(0)
        
        logger.info(f"✅ Preprocessing completed. Final shape: {df.shape}")
        return df
    
    def analyze_relationships(self, df: pd.DataFrame) -> Dict:
        """
        Analyze relationships between news, search, and stock data
        """
        logger.info("📊 Analyzing relationships...")
        
        analysis_results = {}
        
        # 1. Correlation analysis
        analysis_results['correlations'] = self._correlation_analysis(df)
        
        # 2. Causality analysis
        analysis_results['causality'] = self._causality_summary(df)
        
        # 3. Feature importance
        analysis_results['feature_importance'] = self.feature_importance
        
        # 4. Temporal patterns
        analysis_results['temporal_patterns'] = self._temporal_pattern_analysis(df)
        
        return analysis_results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Detailed correlation analysis"""
        
        correlations = {}
        
        # Search-stock correlations
        search_cols = [col for col in df.columns if 'search_change' in col]
        stock_cols = [col for col in df.columns if 'stock_change' in col]
        
        for search_col in search_cols:
            for stock_col in stock_cols:
                if search_col != stock_col:
                    corr, p_value = pearsonr(df[search_col].fillna(0), df[stock_col].fillna(0))
                    correlations[f'{search_col}_vs_{stock_col}'] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return correlations
    
    def _causality_summary(self, df: pd.DataFrame) -> Dict:
        """Summarize causality analysis results"""
        
        causality_summary = {}
        
        # Lead-lag relationships
        lead_lag_cols = [col for col in df.columns if 'leads' in col]
        for col in lead_lag_cols:
            causality_summary[col] = {
                'mean_effect': df[col].mean(),
                'std_effect': df[col].std(),
                'positive_ratio': (df[col] > 0).mean()
            }
        
        return causality_summary
    
    def _temporal_pattern_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns"""
        
        patterns = {}
        
        # Day of week patterns
        if 'day_of_week' in df.columns:
            patterns['day_of_week'] = df.groupby('day_of_week')['combined_spike_score'].mean().to_dict()
        
        # Month patterns
        if 'month' in df.columns:
            patterns['month'] = df.groupby('month')['combined_spike_score'].mean().to_dict()
        
        return patterns

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = EnhancedNewsPreprocessor()
    
    # Example data (replace with your actual data)
    news_df = pd.DataFrame({
        'date': pd.date_range('2014-01-01', '2019-12-31', freq='D'),
        'heading': ['Sample news'] * 2192,
        'content': ['Sample content'] * 2192,
        'sentiment': ['positive'] * 2192
    })
    
    search_df = pd.DataFrame({
        'date': pd.date_range('2014-01-01', '2019-12-31', freq='D'),
        'search_volume': np.random.normal(100, 20, 2192)
    })
    
    stock_df = pd.DataFrame({
        'date': pd.date_range('2014-01-01', '2019-12-31', freq='D'),
        'Close (Rs.)': np.random.normal(100, 5, 2192)
    })
    
    # Run preprocessing pipeline
    processed_df = preprocessor.preprocess_pipeline(news_df, search_df, stock_df)
    
    # Analyze relationships
    analysis = preprocessor.analyze_relationships(processed_df)
    
    print("✅ Enhanced preprocessing completed!")
    print(f"Final dataset shape: {processed_df.shape}")
    print(f"Number of features: {len(processed_df.columns)}") 