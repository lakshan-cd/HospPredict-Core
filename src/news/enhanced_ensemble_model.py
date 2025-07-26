import pandas as pd
import numpy as np
import gc
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleModel:
    """
    Enhanced Ensemble Model for Hotel Stock Price Movement Prediction
    Combines spike prediction and price movement prediction with advanced features
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.spike_model = None
        self.price_models = {}
        self.ensemble_model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def create_advanced_features(self, df):
        """Create advanced features for better prediction"""
        print("🔧 Creating advanced features...")
        
        # 1. Temporal Features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # 2. Text-based Features
        df['text'] = df['heading'].fillna('') + ' ' + df['content'].fillna('')
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
        
        # 3. Sentiment Intensity Features
        df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1}).fillna(0)
        df['sentiment_abs'] = abs(df['sentiment_encoded'])
        
        # 4. Source Credibility Features
        source_credibility = {
            'economynext': 0.9, 'dailynews': 0.8, 'sundaytimes': 0.85,
            'island': 0.75, 'colombopage': 0.7, 'newsfirst': 0.8,
            'adaderana': 0.75, 'hirunews': 0.7, 'lankadeepa': 0.8
        }
        df['source_credibility'] = df['source'].map(source_credibility).fillna(0.5)
        
        # 5. Hotel-specific Features
        df['hotel_group_encoded'] = df['hotel_group'].astype('category').cat.codes
        df['type_encoded'] = df['type'].astype('category').cat.codes
        
        # 6. Market Context Features (rolling averages)
        hotel_columns = [col for col in df.columns if '_change_1d' in col]
        for col in hotel_columns:
            hotel = col.replace('_change_1d', '')
            # Rolling mean of price changes for market context
            df[f'{hotel}_rolling_mean_7d'] = df[col].rolling(7, min_periods=1).mean()
            df[f'{hotel}_rolling_std_7d'] = df[col].rolling(7, min_periods=1).std()
            df[f'{hotel}_volatility'] = df[f'{hotel}_rolling_std_7d'] / abs(df[f'{hotel}_rolling_mean_7d'] + 1e-8)
        
        # 7. Interaction Features
        df['sentiment_source_interaction'] = df['sentiment_encoded'] * df['source_credibility']
        df['type_sentiment_interaction'] = df['type_encoded'] * df['sentiment_encoded']
        
        # 8. News Volume Features
        df['daily_news_count'] = df.groupby('date')['text'].transform('count')
        df['hotel_daily_news_count'] = df.groupby(['date', 'hotel_group'])['text'].transform('count')
        
        # 9. Text Complexity Features
        df['text_complexity'] = df['text'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
        
        return df
    
    def create_price_targets(self, df, threshold=0.005):
        """Create price movement targets with enhanced thresholds"""
        hotel_columns = [col for col in df.columns if '_change_1d' in col]
        
        for col in hotel_columns:
            hotel = col.replace('_change_1d', '')
            
            # Dynamic threshold based on hotel volatility
            hotel_std = df[col].std()
            dynamic_threshold = max(threshold, hotel_std * 0.5)
            
            def create_target(change):
                if pd.isna(change):
                    return np.nan
                if change > dynamic_threshold:
                    return 'up'
                elif change < -dynamic_threshold:
                    return 'down'
                else:
                    return 'neutral'
            
            df[f'{hotel}_target'] = df[col].apply(create_target)
        
        return df
    
    def build_spike_model(self, X_train, y_train, X_test, y_test):
        """Build enhanced spike prediction model"""
        print("🏗️ Building Spike Prediction Model...")
        
        # Multiple base models
        models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=self.random_state),
            'lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'svm': SVC(probability=True, random_state=self.random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state, max_iter=500)
        }
        
        # Ensemble voting classifier
        self.spike_model = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft'
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train model
        self.spike_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.spike_model.predict(X_test)
        y_proba = self.spike_model.predict_proba(X_test)[:, 1]
        
        print("📊 Spike Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        return y_pred, y_proba
    
    def build_price_models(self, df, vectorizer, feature_cols):
        """Build enhanced price movement models for each hotel"""
        print("🏗️ Building Price Movement Models...")
        
        hotel_columns = [col for col in df.columns if '_change_1d' in col]
        price_models = {}
        metrics_summary = []
        
        for col in hotel_columns:
            hotel = col.replace('_change_1d', '')
            target_col = f'{hotel}_target'
            
            if target_col not in df.columns:
                continue
                
            # Filter valid data
            y_price = df[target_col]
            valid_mask = y_price.notna()
            
            if valid_mask.sum() < 30:  # Minimum samples threshold
                print(f"⚠️ Skipping {hotel} — insufficient samples ({valid_mask.sum()})")
                continue
            
            print(f"📈 Processing {hotel} with {valid_mask.sum()} samples...")
            
            # Prepare features
            X_price_text_raw = df.loc[valid_mask, 'text']
            X_price_text = vectorizer.transform(X_price_text_raw)
            
            # Enhanced numerical features
            enhanced_features = feature_cols + [
                'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
                'source_credibility', 'sentiment_source_interaction', 'type_sentiment_interaction',
                'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
                'is_weekend', 'is_month_start', 'is_month_end'
            ]
            
            # Add hotel-specific rolling features
            hotel_rolling_features = [f'{hotel}_rolling_mean_7d', f'{hotel}_rolling_std_7d', f'{hotel}_volatility']
            enhanced_features.extend([f for f in hotel_rolling_features if f in df.columns])
            
            X_price_num = csr_matrix(df.loc[valid_mask, enhanced_features].fillna(0).astype('float32').values)
            X_price_combined = hstack([X_price_text, X_price_num], format='csr')
            
            y_price = y_price[valid_mask]
            
            # Encode labels
            le = LabelEncoder()
            y_price_encoded = le.fit_transform(y_price)
            self.label_encoders[hotel] = le
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_price_combined, y_price_encoded, test_size=0.2, 
                stratify=y_price_encoded, random_state=self.random_state
            )
            
            # Build ensemble model for this hotel
            hotel_models = {
                'rf': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=self.random_state, n_jobs=-1),
                'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
            
            hotel_ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in hotel_models.items()],
                voting='soft'
            )
            
            # Handle class imbalance
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Train model
            hotel_ensemble.fit(X_train_balanced, y_train_balanced)
            price_models[hotel] = hotel_ensemble
            
            # Evaluate
            y_pred = hotel_ensemble.predict(X_test)
            y_proba = hotel_ensemble.predict_proba(X_test)
            
            # Calculate metrics
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            
            metrics_summary.append({
                'hotel': hotel,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'auc_score': auc,
                'samples': len(y_test)
            })
            
            print(f"✅ {hotel}: F1={f1:.3f}, AUC={auc:.3f}")
        
        self.price_models = price_models
        
        # Print summary
        metrics_df = pd.DataFrame(metrics_summary)
        print("\n📊 Price Model Performance Summary:")
        print(metrics_df.round(3))
        
        return metrics_df
    
    def build_ensemble_model(self, df, spike_proba, price_probas):
        """Build final ensemble model combining spike and price predictions"""
        print("🏗️ Building Final Ensemble Model...")
        
        # Combine predictions with original features
        ensemble_features = []
        
        # 1. Spike prediction probability
        ensemble_features.append(spike_proba)
        
        # 2. Price prediction probabilities for each hotel
        for hotel, proba in price_probas.items():
            if proba is not None:
                ensemble_features.extend(proba)
        
        # 3. Original features
        feature_cols = [
            'sentiment_encoded', 'source_credibility', 'type_encoded', 
            'hotel_group_encoded', 'day_of_week', 'month', 'quarter',
            'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
            'sentiment_source_interaction', 'type_sentiment_interaction',
            'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        for col in feature_cols:
            if col in df.columns:
                ensemble_features.append(df[col].values)
        
        # Combine all features
        X_ensemble = np.column_stack(ensemble_features)
        
        # Create target: weighted combination of spike and price movement
        # This is a simplified approach - you might want to customize this
        y_ensemble = (spike_proba > 0.5).astype(int)  # Binary: spike or no spike
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_ensemble, y_ensemble, test_size=0.2, 
            stratify=y_ensemble, random_state=self.random_state
        )
        
        # Final ensemble model
        final_models = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=self.random_state, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=self.random_state),
            'lr': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in final_models.items()],
            voting='soft'
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train model
        self.ensemble_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_test)
        y_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        print("📊 Final Ensemble Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        return y_pred, y_proba
    
    def train(self, df_path):
        """Complete training pipeline"""
        print("🚀 Starting Enhanced Ensemble Training...")
        
        # Load and preprocess data
        df = pd.read_csv(df_path)
        df = self.create_advanced_features(df)
        df = self.create_price_targets(df)
        
        # Prepare text features
        print("📝 Preparing text features...")
        df['text'] = df['heading'].fillna('') + ' ' + df['content'].fillna('')
        
        # Train-test split for spike model
        spike_target = df['spike_linked'] if 'spike_linked' in df.columns else df['linked_to_stock_spike']
        
        X_text_raw_train, X_text_raw_test, y_spike_train, y_spike_test = train_test_split(
            df['text'], spike_target, test_size=0.2, 
            stratify=spike_target, random_state=self.random_state
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_text_train = self.vectorizer.fit_transform(X_text_raw_train)
        X_text_test = self.vectorizer.transform(X_text_raw_test)
        
        # Prepare numerical features for spike model
        spike_feature_cols = [
            'sentiment_encoded', 'source_credibility', 'type_encoded', 
            'hotel_group_encoded', 'day_of_week', 'month', 'quarter',
            'text_length', 'word_count', 'avg_word_length', 'sentiment_abs',
            'sentiment_source_interaction', 'type_sentiment_interaction',
            'daily_news_count', 'hotel_daily_news_count', 'text_complexity',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        X_num_train = csr_matrix(df.loc[X_text_raw_train.index, spike_feature_cols].fillna(0).astype('float32').values)
        X_num_test = csr_matrix(df.loc[X_text_raw_test.index, spike_feature_cols].fillna(0).astype('float32').values)
        
        X_train_combined = hstack([X_text_train, X_num_train], format='csr')
        X_test_combined = hstack([X_text_test, X_num_test], format='csr')
        
        # Build spike model
        spike_pred, spike_proba = self.build_spike_model(X_train_combined, y_spike_train, X_test_combined, y_spike_test)
        
        # Build price models
        price_metrics = self.build_price_models(df, self.vectorizer, spike_feature_cols)
        
        # Get price predictions for ensemble
        price_probas = {}
        for hotel in self.price_models.keys():
            target_col = f'{hotel}_target'
            if target_col in df.columns:
                valid_mask = df[target_col].notna()
                if valid_mask.sum() > 0:
                    X_hotel_text = self.vectorizer.transform(df.loc[valid_mask, 'text'])
                    X_hotel_num = csr_matrix(df.loc[valid_mask, spike_feature_cols].fillna(0).astype('float32').values)
                    X_hotel_combined = hstack([X_hotel_text, X_hotel_num], format='csr')
                    price_probas[hotel] = self.price_models[hotel].predict_proba(X_hotel_combined)
                else:
                    price_probas[hotel] = None
        
        # Build final ensemble
        ensemble_pred, ensemble_proba = self.build_ensemble_model(df, spike_proba, price_probas)
        
        print("✅ Training completed successfully!")
        
        return {
            'spike_metrics': classification_report(y_spike_test, spike_pred, output_dict=True),
            'price_metrics': price_metrics,
            'ensemble_metrics': classification_report(ensemble_pred, ensemble_pred, output_dict=True)
        }
    
    def predict(self, news_text, news_features):
        """Make predictions for new news articles"""
        # Vectorize text
        X_text = self.vectorizer.transform([news_text])
        
        # Prepare features
        X_num = csr_matrix([news_features])
        X_combined = hstack([X_text, X_num], format='csr')
        
        # Get spike prediction
        spike_proba = self.spike_model.predict_proba(X_combined)[0, 1]
        
        # Get price predictions for each hotel
        price_predictions = {}
        for hotel, model in self.price_models.items():
            price_proba = model.predict_proba(X_combined)[0]
            price_predictions[hotel] = {
                'probabilities': price_proba,
                'prediction': self.label_encoders[hotel].inverse_transform([np.argmax(price_proba)])[0]
            }
        
        # Get ensemble prediction
        ensemble_proba = self.ensemble_model.predict_proba(X_combined)[0, 1]
        
        return {
            'spike_probability': spike_proba,
            'price_predictions': price_predictions,
            'ensemble_probability': ensemble_proba
        }
    
    def save_models(self, path):
        """Save all trained models"""
        joblib.dump(self.spike_model, f"{path}/spike_model.pkl")
        joblib.dump(self.price_models, f"{path}/price_models.pkl")
        joblib.dump(self.ensemble_model, f"{path}/ensemble_model.pkl")
        joblib.dump(self.vectorizer, f"{path}/vectorizer.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        print(f"✅ Models saved to {path}")
    
    def load_models(self, path):
        """Load trained models"""
        self.spike_model = joblib.load(f"{path}/spike_model.pkl")
        self.price_models = joblib.load(f"{path}/price_models.pkl")
        self.ensemble_model = joblib.load(f"{path}/ensemble_model.pkl")
        self.vectorizer = joblib.load(f"{path}/vectorizer.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        print(f"✅ Models loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize model
    ensemble_model = EnhancedEnsembleModel(random_state=42)
    
    # Train model (replace with your data path)
    # results = ensemble_model.train('/path/to/your/processed_data.csv')
    
    # Save models
    # ensemble_model.save_models('./models')
    
    print("🎯 Enhanced Ensemble Model Ready!") 