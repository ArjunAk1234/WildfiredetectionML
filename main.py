# Wildfire Detection and Spread Prediction using Machine Learning
# V10: Final version with comprehensive outputs, visualizations, and prediction pipeline.
# V11: Enhanced LSTM evaluation with NRMSE, persistence baseline, and detailed interpretation as requested.
# V12: Integrated user's advanced LSTM implementation with data augmentation and a deeper model architecture, while retaining critical evaluation components.
# V13: Adjusted data augmentation to produce a dataset of ~1000 samples for the LSTM model.
# V15: Made LSTM data augmentation intelligent to handle any input dataset size correctly.
# V16: Added hyperparameter tuning for the best regression model to improve fire spread prediction.
# V17: Added outlier removal for regression training and SHAP for model explainability.
# V28: Replaced manual data augmentation with a dynamic, rule-based generator for creating synthetic low-risk samples.
# V29 (User Request): Modified prediction pipeline to use both Classification and Regression, removing Clustering from the final pipeline logic.
# V30 (Refinement): Ensured no duplicate outputs are generated and the prediction pipeline correctly selects the best models.
# V31 (User Request): Prediction pipeline now uses Classification and the best Clustering model. A K-Nearest Neighbors classifier is trained on cluster labels to enable predictions for new data points.
# V32 (Integration): Merged the adaptive, data-driven risk threshold calibration for a more robust prediction pipeline.
# V33 (User Request): Simplified prediction pipeline to use Classification ONLY with fixed risk thresholds. Training phase remains comprehensive.
# V34 (Final User Request): Re-engineered prediction pipeline for holistic, area-wide risk assessment from entire files, with monthly risk breakdown.
# V35 (Refinement): Streamlined prediction output to remove detailed monthly lists and add a definitive 'safe area' conclusion.
# V36 (Final Guardrail): Integrated a hard-coded rule in the prediction pipeline to override the model for physically impossible fire conditions (e.g., low temperature, high rain), ensuring robust real-world predictions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from kneed import KneeLocator
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class WildfireMLProject:
    def __init__(self):
        self.data = None
        self.X = None
        self.X_pca = None
        self.y_area = None
        self.y_fire = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.models = {}
        self.regression_models = {}
        self.results = {}
        self.feature_names = None
        # Attributes for clustering prediction pipeline (maintained for comprehensive training phase)
        self.cluster_scaler = StandardScaler()
        self.cluster_features = None
        self.cluster_predictor_model = None
        self.best_clustering_name = None

    def load_and_prepare_data(self, file_path):
        """Load data from the provided CSV file path"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully from '{file_path}'!")
            print(f"Original dataset shape: {self.data.shape}")
            print("\nFirst few rows:")
            print(self.data.head())
            return self.data
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
            return None
    
    def augment_data_with_low_risk_samples(self, n_samples=50):
        """
        Dynamically generates and adds synthetic low-risk data to the training set 
        to help the model better identify and classify safe conditions.
        """
        if self.data is None: return
        print("\n" + "="*50)
        print("PERFORMING DYNAMIC DATA AUGMENTATION")
        print("="*50)

        synthetic_samples = []
        # Define rules for low-risk scenarios
        winter_months = ['jan', 'feb', 'dec', 'nov']
        all_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

        for _ in range(n_samples):
            # 50% chance of having some rain
            rain = np.random.choice([0.0, np.random.uniform(0.1, 5.0)], p=[0.5, 0.5])
            
            sample = {
                'X': np.random.randint(1, 10),
                'Y': np.random.randint(1, 10),
                'month': np.random.choice(winter_months),
                'day': np.random.choice(all_days),
                'FFMC': np.random.uniform(70.0, 87.0),
                'DMC': np.random.uniform(2.0, 20.0),
                'DC': np.random.uniform(10.0, 60.0),
                'ISI': np.random.uniform(0.5, 4.0),
                'temp': np.random.uniform(1.0, 10.0),
                'RH': np.random.randint(75, 101),
                'wind': np.random.uniform(0.5, 3.0),
                'rain': rain,
                'area': 0.0  # Crucially, all synthetic samples are non-fire events
            }
            synthetic_samples.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples)
        self.data = pd.concat([self.data, synthetic_df], ignore_index=True)
        
        print(f"Added {len(synthetic_df)} synthetic low-risk samples to the dataset.")
        print(f"New augmented dataset shape: {self.data.shape}")

    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        if self.data is None: return
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        print("\nDataset Info:"); self.data.info(verbose=False)
        print("\nDescriptive Statistics:"); print(self.data.describe())
        print("\nMissing Values:"); print(self.data.isnull().sum())
        fire_count = (self.data['area'] > 0).sum()
        no_fire_count = (self.data['area'] == 0).sum()
        print(f"\nFire Distribution:\nFire incidents: {fire_count}\nNo fire incidents: {no_fire_count}\nFire rate: {fire_count/(fire_count + no_fire_count)*100:.2f}%")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        sns.histplot(np.log1p(self.data['area']), bins=30, kde=True, ax=axes[0,0], color='orange')
        axes[0,0].set_title('Log(Area+1) Distribution')
        weather_cols = ['temp', 'RH', 'wind', 'rain']; corr_data = self.data[weather_cols + ['area']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[0,1]); axes[0,1].set_title('Weather Conditions Correlation')
        fire_indices = ['FFMC', 'DMC', 'DC', 'ISI']; corr_fire = self.data[fire_indices + ['area']].corr()
        sns.heatmap(corr_fire, annot=True, cmap='coolwarm', center=0, ax=axes[0,2]); axes[0,2].set_title('Fire Weather Indices Correlation')
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        sns.countplot(data=self.data, x='month', order=month_order, ax=axes[1,0]); axes[1,0].set_title('Fire Incidents by Month'); axes[1,0].tick_params(axis='x', rotation=45)
        day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        sns.countplot(data=self.data, x='day', order=day_order, ax=axes[1,1]); axes[1,1].set_title('Fire Incidents by Day of Week'); axes[1,1].tick_params(axis='x', rotation=45)
        scatter = axes[1,2].scatter(self.data['X'], self.data['Y'], c=np.log1p(self.data['area']), cmap='YlOrRd', alpha=0.6, s=self.data['area']*5 + 20)
        axes[1,2].set_title('Spatial Distribution of Fires (Log Scale)'); plt.colorbar(scatter, ax=axes[1,2], label='Log(Area+1)')
        plt.tight_layout(); plt.show()

    def data_preprocessing(self):
        """Clean and prepare data for modeling by encoding and scaling features."""
        if self.data is None: return
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        self.data = self.data.dropna()
        self.data['month'] = self.data['month'].astype('category'); self.data['day'] = self.data['day'].astype('category')
        
        # Ensure all possible months and days are in the categories before encoding
        all_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        all_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        self.data['month'] = self.data['month'].cat.set_categories(all_months)
        self.data['day'] = self.data['day'].cat.set_categories(all_days)
        
        self.data['month_encoded'] = self.data['month'].cat.codes; self.data['day_encoded'] = self.data['day'].cat.codes
        self.data['temp_rh_interaction'] = self.data['temp'] * self.data['RH']
        self.data['wind_temp_interaction'] = self.data['wind'] * self.data['temp']
        self.data['fire_weather_index'] = (self.data['FFMC'] + self.data['DMC'] + self.data['DC'] + self.data['ISI']) / 4
        self.data['fire_occurrence'] = (self.data['area'] > 0).astype(int)
        self.data['log_area'] = np.log1p(self.data['area'])
        feature_cols = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month_encoded', 'day_encoded', 'temp_rh_interaction', 'wind_temp_interaction', 'fire_weather_index']
        self.feature_names = feature_cols
        self.X = self.data[feature_cols]
        self.y_area = self.data['log_area']; self.y_fire = self.data['fire_occurrence']
        print(f"Features shape: {self.X.shape}\nTarget (log area) shape: {self.y_area.shape}\nTarget (fire occurrence) shape: {self.y_fire.shape}")
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("All features have been standardized using StandardScaler.")
        self.pca = PCA(n_components=0.95); self.X_pca = self.pca.fit_transform(self.X_scaled)
        print(f"Original features: {self.X.shape[1]}\nPCA components: {self.X_pca.shape[1]}\nExplained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

    def fire_spread_prediction_regression(self):
        """Implement regression models for fire spread prediction using log(area)"""
        if self.data is None: return
        print("\n" + "="*50); print("FIRE SPREAD PREDICTION (REGRESSION)"); print("="*50)
        fire_mask = self.data['area'] > 0
        X_fire = self.X_pca[fire_mask]
        y_fire_area_log = self.y_area[fire_mask]
        if len(X_fire) < 10: print("Insufficient fire incidents found for regression analysis"); return
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(X_fire, y_fire_area_log, test_size=0.2, random_state=42)
        model_definitions = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), 'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42), 'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1), 'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)}
        regression_results = {}
        for name, model in model_definitions.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_reg, self.y_train_reg)
            self.regression_models[name] = model
            y_pred_train_log = model.predict(self.X_train_reg); y_pred_test_log = model.predict(self.X_test_reg)
            train_rmse = np.sqrt(mean_squared_error(self.y_train_reg, y_pred_train_log)); test_rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred_test_log))
            train_r2 = r2_score(self.y_train_reg, y_pred_train_log); test_r2 = r2_score(self.y_test_reg, y_pred_test_log)
            # Guard against empty test set for original scale conversion
            if len(self.y_test_reg) > 0:
                y_pred_test_original = np.expm1(y_pred_test_log); y_test_original = np.expm1(self.y_test_reg)
                test_rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
            else:
                test_rmse_original = np.nan
            print(f"Train RMSE (log): {train_rmse:.3f}, Test RMSE (log): {test_rmse:.3f}")
            print(f"Test RMSE (original scale): {test_rmse_original:.3f}")
            print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
            regression_results[name] = {'Train RMSE (log)': train_rmse, 'Test RMSE (log)': test_rmse, 'Test RMSE (original)': test_rmse_original, 'Train R²': train_r2, 'Test R²': test_r2}
        self.results['regression'] = regression_results
        if len(self.y_test_reg) > 0:
            self.visualize_regression_results(regression_results, self.y_test_reg, self.X_test_reg)

    def hyperparameter_tuning_regression(self):
        """Perform hyperparameter tuning for the best regression model based on RMSE and R-squared."""
        if not hasattr(self, 'X_train_reg') or len(self.X_train_reg) == 0:
            print("Regression training data not found or is empty. Skipping tuning.")
            return
        print("\n" + "="*50); print("HYPERPARAMETER TUNING (REGRESSION)"); print("="*50)
        best_model_name = None
        candidates = []
        for name, result in self.results['regression'].items():
            if result.get('Test R²') is not None and result['Test R²'] > 0.5:
                candidates.append((name, result['Test RMSE (original)']))
        if candidates:
            best_model_name = min(candidates, key=lambda item: item[1])[0]
            print(f"Found model(s) with R² > 0.5. Choosing '{best_model_name}' for tuning due to lowest RMSE.")
        else:
            fallback_candidates = []
            for name, result in self.results['regression'].items():
                if result.get('Test RMSE (original)') is not None:
                     fallback_candidates.append((name, result['Test RMSE (original)']))
            if fallback_candidates:
                best_model_name = min(fallback_candidates, key=lambda item: item[1])[0]
                print(f"Warning: No model had Test R² > 0.5. Falling back to tuning '{best_model_name}' as it has the lowest RMSE.")
        if not best_model_name or best_model_name not in ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM']:
            print(f"Selected model '{best_model_name}' is not suitable for the defined tuning. Skipping.")
            return
        print(f"Tuning the best performing model: {best_model_name}")
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
        model_to_tune = self.regression_models[best_model_name]
        grid_search = RandomizedSearchCV(estimator=model_to_tune, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error', random_state=42)
        grid_search.fit(self.X_train_reg, self.y_train_reg)
        print(f"\nBest parameters for {best_model_name}: {grid_search.best_params_}")
        tuned_model = grid_search.best_estimator_
        tuned_model_name = f"Tuned {best_model_name}"
        self.regression_models[tuned_model_name] = tuned_model
        y_pred_test_log = tuned_model.predict(self.X_test_reg)
        test_rmse_log = np.sqrt(mean_squared_error(self.y_test_reg, y_pred_test_log))
        test_r2 = r2_score(self.y_test_reg, y_pred_test_log)
        y_pred_test_original = np.expm1(y_pred_test_log); y_test_original = np.expm1(self.y_test_reg)
        test_rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
        print(f"\nTuned {best_model_name} Performance:")
        print(f"Test RMSE (log): {test_rmse_log:.3f}\nTest RMSE (original scale): {test_rmse_original:.3f}\nTest R²: {test_r2:.3f}")
        self.results['regression'][tuned_model_name] = {'Test RMSE (log)': test_rmse_log, 'Test RMSE (original)': test_rmse_original, 'Test R²': test_r2, 'Train RMSE (log)': np.nan, 'Train R²': np.nan}

    def fire_detection_classification(self):
        """Implement classification models for fire detection"""
        if self.data is None: return
        print("\n" + "="*50); print("FIRE DETECTION (CLASSIFICATION)"); print("="*50)
        self.X_train_cls, self.X_test_cls, self.y_train_cls, self.y_test_cls = train_test_split(self.X_pca, self.y_fire, test_size=0.2, random_state=42, stratify=self.y_fire)
        classification_models = {'Logistic Regression': LogisticRegression(random_state=42), 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'), 'SVM': SVC(random_state=42, probability=True, class_weight='balanced'), 'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(self.y_fire) - sum(self.y_fire))/sum(self.y_fire))}
        classification_results = {}
        for name, model in classification_models.items():
            print(f"\nTraining {name}..."); model.fit(self.X_train_cls, self.y_train_cls)
            y_pred_test = model.predict(self.X_test_cls)
            classification_results[name] = {'Train Accuracy': accuracy_score(self.y_train_cls, model.predict(self.X_train_cls)), 'Test Accuracy': accuracy_score(self.y_test_cls, y_pred_test), 'Precision': precision_score(self.y_test_cls, y_pred_test, zero_division=0), 'Recall': recall_score(self.y_test_cls, y_pred_test, zero_division=0), 'F1-Score': f1_score(self.y_test_cls, y_pred_test, zero_division=0)}
            print(f"Test Accuracy: {classification_results[name]['Test Accuracy']:.3f}, Precision: {classification_results[name]['Precision']:.3f}\nRecall: {classification_results[name]['Recall']:.3f}, F1-Score: {classification_results[name]['F1-Score']:.3f}")
        self.results['classification'] = classification_results; self.models['classification'] = classification_models
        self.visualize_classification_results(classification_results, self.X_test_cls, self.y_test_cls, classification_models)
    
    def improved_clustering_analysis(self):
        """Implement and compare clustering algorithms on a targeted feature set."""
        if self.data is None: return
        print("\n" + "="*50); print("FIRE HOTSPOT CLUSTERING ANALYSIS"); print("="*50)
        self.cluster_features = ['X', 'Y', 'fire_weather_index', 'temp']
        if not all(feat in self.data.columns for feat in self.cluster_features): return
        
        data_for_clustering = self.data.reset_index(drop=True)
        X_cluster = data_for_clustering[self.cluster_features]
        X_cluster_scaled = self.cluster_scaler.fit_transform(X_cluster)
        
        clustering_methods = {'KMeans': KMeans(n_clusters=3, random_state=42, n_init=10), 'DBSCAN': DBSCAN(eps=0.8, min_samples=5), 'Agglomerative': AgglomerativeClustering(n_clusters=3)}
        def evaluate_clustering(X, labels):
            return {'silhouette': silhouette_score(X, labels), 'calinski_harabasz': calinski_harabasz_score(X, labels)} if len(set(labels)) > 1 else {'silhouette': -1, 'calinski_harabasz': -1}
        
        clustering_results = {}
        print("Applying and evaluating clustering algorithms...")
        for name, method in clustering_methods.items():
            labels = method.fit_predict(X_cluster_scaled)
            metrics = evaluate_clustering(X_cluster_scaled, labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"\n{name} Results:\n  Clusters found: {n_clusters}\n  Silhouette Score: {metrics['silhouette']:.3f}\n  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f}")
            clustering_results[name] = {'labels': labels, 'n_clusters': n_clusters, **metrics}
        self.results['clustering'] = clustering_results
        self.visualize_clustering_results()

    def hyperparameter_tuning_clustering(self):
        """Find and apply the optimal epsilon for DBSCAN."""
        if self.cluster_features is None:
            print("Clustering data not found. Run `improved_clustering_analysis` first.")
            return
            
        print("\n" + "="*50); print("HYPERPARAMETER TUNING (DBSCAN)"); print("="*50)
        data_for_clustering = self.data.reset_index(drop=True)
        X_cluster_scaled = self.cluster_scaler.transform(data_for_clustering[self.cluster_features])

        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(X_cluster_scaled)
        distances, indices = neighbors_fit.kneighbors(X_cluster_scaled)
        distances = np.sort(distances, axis=0)[:, 4]
        kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
        optimal_eps = kneedle.elbow_y
        if optimal_eps:
            print(f"Found optimal epsilon for DBSCAN: {optimal_eps:.3f}")
            tuned_dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
            labels = tuned_dbscan.fit_predict(X_cluster_scaled)
            metrics = {'silhouette': silhouette_score(X_cluster_scaled, labels) if len(set(labels)) > 1 else -1}
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Tuned DBSCAN Results:\n  Clusters found: {n_clusters}\n  Silhouette Score: {metrics['silhouette']:.3f}")
            self.results['clustering']['Tuned DBSCAN'] = {'labels': labels, 'n_clusters': n_clusters, **metrics, 'calinski_harabasz': np.nan}
            self.visualize_clustering_results()
        else:
            print("Could not automatically determine optimal epsilon for DBSCAN.")

    def train_cluster_predictor(self):
        """Trains a KNN classifier on the results of the best clustering algorithm."""
        print("\n" + "="*50); print("TRAINING CLUSTER PREDICTOR MODEL"); print("="*50)
        if 'clustering' not in self.results or not self.results['clustering']:
            print("Clustering results not found. Skipping training of cluster predictor.")
            return

        if 'Tuned DBSCAN' in self.results['clustering']:
            self.best_clustering_name = 'Tuned DBSCAN'
        else:
            valid_scores = {name: result['silhouette'] for name, result in self.results['clustering'].items() if result['silhouette'] != -1}
            if not valid_scores:
                print("No valid clustering results to build a predictor from.")
                return
            self.best_clustering_name = max(valid_scores, key=valid_scores.get)
        
        print(f"Using '{self.best_clustering_name}' results to train the cluster predictor.")
        
        best_labels = self.results['clustering'][self.best_clustering_name]['labels']
        data_for_clustering = self.data.reset_index(drop=True)
        X_cluster_scaled = self.cluster_scaler.transform(data_for_clustering[self.cluster_features])
        
        train_mask = best_labels != -1
        X_train_predictor = X_cluster_scaled[train_mask]
        y_train_predictor = best_labels[train_mask]

        if len(y_train_predictor) < 5:
            print("Not enough non-noise points to train a reliable cluster predictor.")
            return

        self.cluster_predictor_model = KNeighborsClassifier(n_neighbors=5)
        self.cluster_predictor_model.fit(X_train_predictor, y_train_predictor)
        print("Successfully trained a K-Nearest Neighbors model to predict cluster IDs.")

    def visualize_regression_results(self, results, y_test, X_test):
        """Visualize regression results"""
        if len(y_test) == 0:
            print("No regression test data to visualize.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle("Regression Model Performance", fontsize=16)
        model_names = list(results.keys()); test_rmse_log = [results[name]['Test RMSE (log)'] for name in model_names]
        sns.barplot(x=model_names, y=test_rmse_log, ax=axes[0,0]); axes[0,0].set_title('Test RMSE Comparison (Log Scale)'); axes[0,0].set_ylabel('RMSE (log)'); axes[0,0].tick_params(axis='x', rotation=45)
        test_r2 = [results[name]['Test R²'] for name in model_names]
        sns.barplot(x=model_names, y=test_r2, ax=axes[0,1]); axes[0,1].set_title('Test R² Comparison'); axes[0,1].set_ylabel('R² Score'); axes[0,1].tick_params(axis='x', rotation=45)
        best_model_name = min(results, key=lambda x: results[x]['Test RMSE (log)']); best_model = self.regression_models[best_model_name]
        y_pred_best = best_model.predict(X_test)
        axes[1,0].scatter(y_test, y_pred_best, alpha=0.6); axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2); axes[1,0].set_xlabel('Actual Log(Area+1)'); axes[1,0].set_ylabel('Predicted Log(Area+1)'); axes[1,0].set_title(f'Actual vs Predicted ({best_model_name})')
        residuals = y_test - y_pred_best
        sns.histplot(residuals, kde=True, ax=axes[1,1]); axes[1,1].set_xlabel('Residuals'); axes[1,1].set_title('Residuals Distribution')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    def visualize_classification_results(self, results, X_test, y_test, models):
        """Visualize classification results with a corrected feature importance plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Classification Model Performance", fontsize=16)

        results_df = pd.DataFrame(results).T.sort_values('F1-Score', ascending=False)
        results_df.plot(kind='bar', y=['Test Accuracy', 'Precision', 'Recall', 'F1-Score'], ax=axes[0,0])
        axes[0,0].set_title('Classification Metrics Comparison'); axes[0,0].set_ylabel('Score'); axes[0,0].tick_params(axis='x', rotation=45)

        best_model_name = results_df.index[0]; best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test); cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix ({best_model_name})'); axes[0,1].set_xlabel('Predicted'); axes[0,1].set_ylabel('Actual')
        
        if hasattr(best_model, 'feature_importances_'):
            pca_importances = best_model.feature_importances_
            pca_feature_names = [f'PCA_Component_{i}' for i in range(len(pca_importances))]
            importance_df = pd.DataFrame({'feature': pca_feature_names, 'importance': pca_importances}).sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[1,0])
            axes[1,0].set_title(f'Principal Component Importance ({best_model_name})')
            
            if hasattr(best_model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                y_proba = best_model.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, y_proba); roc_auc = auc(fpr, tpr)
                axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[1,1].set_xlim([0.0, 1.0]); axes[1,1].set_ylim([0.0, 1.05])
                axes[1,1].set_xlabel('False Positive Rate'); axes[1,1].set_ylabel('True Positive Rate')
                axes[1,1].set_title('ROC Curve'); axes[1,1].legend(loc="lower right")
        else:
            axes[1,0].remove(); axes[1,1].remove()
            ax_bottom_centered = fig.add_subplot(2, 1, 2)
            if hasattr(best_model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                y_proba = best_model.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, y_proba); roc_auc = auc(fpr, tpr)
                ax_bottom_centered.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_bottom_centered.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_bottom_centered.set_xlim([0.0, 1.0]); ax_bottom_centered.set_ylim([0.0, 1.05])
                ax_bottom_centered.set_xlabel('False Positive Rate'); ax_bottom_centered.set_ylabel('True Positive Rate')
                ax_bottom_centered.set_title('ROC Curve'); ax_bottom_centered.legend(loc="lower right")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    def visualize_clustering_results(self):
        """Visualize clustering results for all tested algorithms side-by-side."""
        if self.data is None: return
        if 'clustering' not in self.results or not self.results['clustering']: return
        n_methods = len(self.results['clustering'])
        if n_methods == 0: return
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), sharex=True, sharey=True)
        fig.suptitle('Clustering Algorithm Comparison on Spatial & Weather Data', fontsize=16)
        if n_methods == 1: axes = [axes]
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        for i, (name, result) in enumerate(self.results['clustering'].items()):
            ax = axes[i]; labels, is_noise = result['labels'], (result['labels'] == -1)
            df_for_plot = self.data.reset_index(drop=True)

            scatter_clusters = ax.scatter(df_for_plot.loc[~is_noise, 'X'], df_for_plot.loc[~is_noise, 'Y'], c=labels[~is_noise], cmap='viridis', alpha=0.8, s=np.log1p(df_for_plot.loc[~is_noise, 'area'])*20 + 20)
            if np.sum(is_noise) > 0: ax.scatter(df_for_plot.loc[is_noise, 'X'], df_for_plot.loc[is_noise, 'Y'], c='gray', alpha=0.5, s=15, marker='x', label='Noise')
            ax.set_title(f"{name}\n(Clusters: {result['n_clusters']}, Silhouette: {result.get('silhouette', 'N/A'):.2f})")
            ax.set_xlabel('X Coordinate')
            if i == 0: ax.set_ylabel('Y Coordinate')
            ax.grid(True, linestyle='--', alpha=0.6)
            if np.sum(is_noise) > 0: ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
        
    def generate_performance_report(self):
        """Generate a comprehensive performance report for all project stages."""
        print("\n" + "="*60); print("COMPREHENSIVE PERFORMANCE REPORT"); print("="*60)
        print("\n1. FIRE SPREAD PREDICTION (REGRESSION) - Evaluated on original 'area' scale"); print("-" * 60)
        if 'regression' in self.results: print(pd.DataFrame(self.results['regression']).T.sort_values('Test RMSE (log)'))
        print("\n2. FIRE DETECTION (CLASSIFICATION)"); print("-" * 60)
        if 'classification' in self.results: print(pd.DataFrame(self.results['classification']).T.sort_values('F1-Score', ascending=False))
        print("\n3. FIRE HOTSPOT CLUSTERING"); print("-" * 60)
        if 'clustering' in self.results:
            print("  Comparison of different clustering algorithms on targeted features (X, Y, FWI, temp):")
            clustering_df = pd.DataFrame(self.results['clustering']).T
            print(clustering_df[['n_clusters', 'silhouette', 'calinski_harabasz']].sort_values('silhouette', ascending=False))

    def hyperparameter_tuning_classification(self):
        """Perform hyperparameter tuning for the RandomForestClassifier."""
        if self.data is None: return
        print("\n" + "="*50); print("HYPERPARAMETER TUNING (RANDOM FOREST CLASSIFIER)"); print("="*50)
        rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        rf_grid.fit(self.X_pca, self.y_fire)
        print(f"\nBest RF parameters found: {rf_grid.best_params_}")
        print(f"Best RF F1-score from GridSearch: {rf_grid.best_score_:.3f}")
        self.models['classification']['Tuned Random Forest'] = rf_grid.best_estimator_

    def cross_validation_analysis(self):
        """Perform 5-fold cross-validation on key classification models."""
        if self.data is None: return
        print("\n" + "="*50); print("CROSS-VALIDATION ANALYSIS"); print("="*50)
        cv_models = {'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), 'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'), 'SVM': SVC(random_state=42)}
        for name, model in cv_models.items():
            scores = cross_val_score(model, self.X_pca, self.y_fire, cv=5, scoring='f1_weighted')
            print(f"\n{name} 5-Fold CV F1-Score: {scores.mean():.3f} (std: {scores.std():.3f})")
    
    def save_pipeline_components(self, filepath="wildfire_pipeline.joblib"):
        """Saves the entire project object, which includes models, scalers, and other necessary components."""
        if self.data is None: return
        print("\n" + "="*50); print("SAVING PREDICTION PIPELINE"); print("="*50)
        print("Saving the entire project state for robust prediction...")
        joblib.dump(self, filepath)
        print(f"Prediction pipeline successfully saved to '{filepath}'")

    def run_full_project(self, file_path):
        """Orchestrates the entire ML project pipeline from a file path."""
        print("Starting Wildfire Detection and Spread Prediction Analysis..."); print("="*60)
        loaded_data = self.load_and_prepare_data(file_path)
        if loaded_data is None:
            print("Project terminated due to data loading failure.")
            return
        
        self.augment_data_with_low_risk_samples()
        self.exploratory_data_analysis()
        self.data_preprocessing()
        self.fire_spread_prediction_regression()
        self.hyperparameter_tuning_regression()
        self.fire_detection_classification()
        self.hyperparameter_tuning_classification()
        self.improved_clustering_analysis()
        self.hyperparameter_tuning_clustering()
        self.train_cluster_predictor() 
        self.generate_performance_report()
        self.cross_validation_analysis()
        self.save_pipeline_components()
        print("\n" + "="*60); print("COMPLETE WILDFIRE ML PROJECT FINISHED!"); print("="*60)

class WildfirePredictor:
    def __init__(self, model_path="wildfire_pipeline.joblib"):
        self.pipeline = None
        self.classification_model = None
        self.scaler = None
        self.pca = None
        self.feature_names = None
        # User-defined fixed thresholds
        self.moderate_threshold = 0.55
        self.high_threshold = 0.74
        self.extreme_threshold = 0.95
        self.month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        self.day_categories = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        self._load_pipeline(model_path)

    def _load_pipeline(self, filepath):
        try:
            self.pipeline = joblib.load(filepath)
            class_models = self.pipeline.models.get('classification', {})
            self.classification_model = class_models.get('Tuned Random Forest', class_models.get('Random Forest', class_models.get('XGBoost')))
            self.scaler = self.pipeline.scaler
            self.pca = self.pipeline.pca
            self.feature_names = self.pipeline.feature_names
            print("WildfirePredictor loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at '{filepath}'. Please run the main analysis first.")
            self.classification_model = None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            self.classification_model = None

    def predict_area_risk(self, prediction_filepath, area_name):
        """Analyzes an entire file representing an area to provide a holistic risk assessment."""
        if self.classification_model is None:
            print(f"Model not loaded. Cannot make predictions for {area_name}.")
            return
        
        try:
            new_data_df = pd.read_csv(prediction_filepath)
        except FileNotFoundError:
            print(f"Error: Prediction file '{prediction_filepath}' not found for {area_name}."); return
        
        # --- Preprocessing for Classification ---
        new_data_df['month'] = pd.Categorical(new_data_df['month'], categories=self.month_categories, ordered=True)
        new_data_df['day'] = pd.Categorical(new_data_df['day'], categories=self.day_categories, ordered=True)
        
        new_data_df['month_encoded'] = new_data_df['month'].cat.codes
        new_data_df['day_encoded'] = new_data_df['day'].cat.codes
        new_data_df['temp_rh_interaction'] = new_data_df['temp'] * new_data_df['RH']
        new_data_df['wind_temp_interaction'] = new_data_df['wind'] * new_data_df['temp']
        new_data_df['fire_weather_index'] = (new_data_df['FFMC'] + new_data_df['DMC'] + new_data_df['DC'] + new_data_df['ISI']) / 4

        # --- Rule-Based Override for Physically Impossible Conditions ---
        # Define conditions where fire is highly improbable
        no_go_mask = (new_data_df['temp'] < 5.0) | (new_data_df['rain'] > 0.1)
        go_mask = ~no_go_mask
        
        # Initialize probabilities array with zeros
        probabilities = np.zeros(len(new_data_df))

        # Only predict on data that passes the rule-based check
        if go_mask.any():
            df_to_predict = new_data_df[go_mask]
            
            df_processed_cls = df_to_predict.reindex(columns=self.feature_names, fill_value=0)
            X_scaled_cls = self.scaler.transform(df_processed_cls)
            X_pca_cls = self.pca.transform(X_scaled_cls)
            
            model_probs = self.classification_model.predict_proba(X_pca_cls)[:, 1]
            probabilities[go_mask] = model_probs
        
        new_data_df['probability'] = probabilities
        
        # --- Holistic Area Analysis ---
        avg_risk_prob = new_data_df['probability'].mean()
        max_risk_prob = new_data_df['probability'].max()
        
        monthly_risk = new_data_df.groupby('month')['probability'].mean().sort_values(ascending=False)
        
        print("\n" + "="*20 + f" RISK ASSESSMENT FOR {area_name.upper()} " + "="*20)
        
        print(f"\n> Average Fire Risk Probability: {avg_risk_prob:.2%}")
        print(f"> Peak Fire Risk Probability Found: {max_risk_prob:.2%}")

        # --- REFINED MONTHLY AND OVERALL CONCLUSION LOGIC ---
        if not monthly_risk.empty and monthly_risk.max() < self.moderate_threshold:
            print("\n> Overall Conclusion:")
            print("  This area is considered safe. The conditions across all analyzed months show no significant chance of wildfire.")
        
        else:
            print("\n> Monthly Risk Profile:")
            if not monthly_risk.empty:
                 print("  - Highest risk months:", ", ".join(monthly_risk.head(2).index.str.capitalize()))
            else:
                print("  - No monthly data available to analyze.")
            
            print("\n> Overall Conclusion:")
            if avg_risk_prob > self.high_threshold or max_risk_prob > self.extreme_threshold:
                print("  This area exhibits a HIGH potential for wildfire. Conditions are frequently favorable for fire starts and spread, especially in peak months.")
            else:
                print("  This area shows a MODERATE potential for wildfire. While not consistently at high risk, there are periods where conditions can become dangerous.")
        
        print("="*60)

    def _categorize_risk(self, probability):
        """Assigns a risk category based on the fixed probability score."""
        if probability < self.moderate_threshold: return 'Low'
        elif probability < self.high_threshold: return 'Moderate'
        elif probability < self.extreme_threshold: return 'High'
        else: return 'Extreme'


if __name__ == "__main__":
    # --- Step 1: Run the full analysis and training pipeline ---
    training_file_path = 'forestfires3.csv'
    if not os.path.exists(training_file_path):
        print(f"FATAL ERROR: Training file '{training_file_path}' not found. Cannot proceed.")
    else:
        project = WildfireMLProject()
        project.run_full_project(training_file_path)

        # --- Step 2: Demonstrate the production-ready prediction pipeline ---
        print("\n" + "="*60)
        print("DEMONSTRATING AREA-WIDE PREDICTION PIPELINE")
        print("="*60)
        
        # --- User must provide these files for prediction ---
        prediction_file_1 = 'forestfirespre3.csv'
        prediction_file_2 = 'forestfirespre5.csv'

        predictor = WildfirePredictor(model_path="wildfire_pipeline.joblib")
        if predictor.classification_model:
            # Predict for the first area
            if os.path.exists(prediction_file_1):
                predictor.predict_area_risk(prediction_file_1, area_name="Area 1")
            else:
                print(f"\nWARNING: Prediction file '{prediction_file_1}' not found. Skipping assessment for Area 1.")
                print("Please create this file with data to test the predictor.")

            # Predict for the second area
            if os.path.exists(prediction_file_2):
                predictor.predict_area_risk(prediction_file_2, area_name="Area 2")
            else:
                print(f"\nWARNING: Prediction file '{prediction_file_2}' not found. Skipping assessment for Area 2.")
                print("Please create this file with data to test the predictor.")
        else:
            print("\nPrediction failed because the predictor could not be initialized.")
