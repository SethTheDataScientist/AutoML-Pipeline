import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import joblib
import time
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Define the AutoMLPipeline class (same as provided)
class AutoMLPipeline:
    def __init__(self, problem_type='auto', output_dir='automl_output'):
        """
        Initialize the AutoML pipeline.
        
        Parameters:
        -----------
        problem_type : str, optional (default='auto')
            The type of problem to solve. Can be 'auto', 'classification', or 'regression'.
        output_dir : str, optional (default='automl_output')
            Directory to save outputs.
        """
        self.problem_type = problem_type
        self.output_dir = output_dir
        self.numeric_features = None
        self.categorical_features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_score = -np.inf if problem_type == 'classification' else np.inf
        self.best_params = None
        self.feature_importances = None
        self.prediction_results = None
        self.metrics = None
        self.preprocessor = None
        self.execution_time = None
        self.model_rankings = None
        self.best_model_name = None
        self.eda_results = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, 'plots'))
            os.makedirs(os.path.join(output_dir, 'models'))
            os.makedirs(os.path.join(output_dir, 'reports'))
    
    def ingest_data(self, data_path, target_column, id_columns=None, drop_columns=None):
        """
        Load and prepare the dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the data file (CSV, Excel, etc.)
        target_column : str
            Name of the target column
        id_columns : list, optional
            List of ID columns to exclude from analysis
        drop_columns : list, optional
            Additional columns to drop
        """
        print(f"[INFO] Ingesting data from {data_path}")
        start_time = time.time()
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(data_path)
        elif data_path.endswith('.parquet'):
            self.data = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV, Excel, or Parquet file.")
        
        self.target = target_column
        
        # Drop columns if specified
        columns_to_drop = []
        if id_columns:
            columns_to_drop.extend(id_columns)
        if drop_columns:
            columns_to_drop.extend(drop_columns)
        
        if columns_to_drop:
            self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
        
        # Separate features and target
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        
        # Auto-detect problem type if not specified
        if self.problem_type == 'auto':
            if self.y.dtype == 'object' or pd.api.types.is_categorical_dtype(self.y) or len(self.y.unique()) <= 10:
                self.problem_type = 'classification'
                print(f"[INFO] Detected problem type: Classification with {len(self.y.unique())} classes")
            else:
                self.problem_type = 'regression'
                print(f"[INFO] Detected problem type: Regression")
        
        # Identify numeric and categorical features
        self.numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        print(f"[INFO] Data ingestion completed in {time.time() - start_time:.2f} seconds")
        print(f"[INFO] Dataset shape: {self.data.shape}")
        print(f"[INFO] Number of numerical features: {len(self.numeric_features)}")
        print(f"[INFO] Number of categorical features: {len(self.categorical_features)}")
        
        return self
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        print("\n[INFO] Starting Exploratory Data Analysis")
        start_time = time.time()
        
        self.eda_results = {}
        
        # Basic statistics
        self.eda_results['basic_stats'] = {
            'data_shape': self.data.shape,
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'null_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
        }
        
        # Target distribution
        plt.figure(figsize=(10, 6))
        if self.problem_type == 'classification':
            value_counts = self.y.value_counts()
            ax = value_counts.plot(kind='bar')
            plt.title('Target Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'target_distribution.png'))
            plt.close()
            
            self.eda_results['target_distribution'] = value_counts.to_dict()
        else:
            plt.hist(self.y, bins=30)
            plt.title('Target Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'target_distribution.png'))
            plt.close()
            
            self.eda_results['target_distribution'] = {
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'mean': float(self.y.mean()),
                'median': float(self.y.median()),
                'std': float(self.y.std())
            }
        
        # Correlation matrix for numeric features
        if len(self.numeric_features) > 0:
            numeric_data = self.data[self.numeric_features + [self.target]]
            correlation_matrix = numeric_data.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'correlation_matrix.png'))
            plt.close()
            
            self.eda_results['correlation_with_target'] = correlation_matrix[self.target].drop(self.target).to_dict()
        
        # Analyze categorical features
        if len(self.categorical_features) > 0:
            cat_stats = {}
            for feature in self.categorical_features:
                value_counts = self.data[feature].value_counts()
                cat_stats[feature] = {
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    'most_common_percentage': float(value_counts.iloc[0] / len(self.data) * 100) if not value_counts.empty else 0
                }
            
            self.eda_results['categorical_features_stats'] = cat_stats
        
        # Missing values analysis
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            plt.figure(figsize=(10, 6))
            missing_data.plot(kind='bar')
            plt.title('Missing Values')
            plt.xlabel('Feature')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'missing_values.png'))
            plt.close()
        
        # Save EDA results
        with open(os.path.join(self.output_dir, 'reports', 'eda_results.json'), 'w') as f:
            json.dump(self.eda_results, f, indent=4, default=str)
        
        print(f"[INFO] EDA completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def prepare_data(self, test_size=0.2, random_state=42, cv_folds=5, use_cv=False):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test split.
        random_state : int, optional (default=42)
            Random seed for reproducibility.
        cv_folds : int, optional (default=5)
            Number of folds to use in cross-validation.
        use_cv : bool, optional (default=False)
            Whether to use cross-validation instead of a simple train-test split.
        """
        print("\n[INFO] Preparing data for modeling")
        start_time = time.time()
        
        # Store cv parameters
        self.cv_folds = cv_folds
        self.use_cv = use_cv
        
        # If using cross-validation, we'll still create a train-test split for final evaluation
        # but the model training will use cross-validation on the training portion
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y if self.problem_type == 'classification' else None
        )
        
        # Store additional info if using cross-validation
        if use_cv:
            print(f"[INFO] Using {cv_folds}-fold cross-validation for model training")
            self.cv_results = {}
        
        print(f"[INFO] Training set shape: {self.X_train.shape}")
        print(f"[INFO] Testing set shape: {self.X_test.shape}")
        print(f"[INFO] Data preparation completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def create_preprocessing_pipeline(self, pca_components=None, feature_selection=None):
        """
        Create a preprocessing pipeline for feature transformation.
        
        Parameters:
        -----------
        pca_components : int, optional
            Number of PCA components to keep. If None, PCA is not applied.
        feature_selection : int, optional
            Number of features to select. If None, feature selection is not applied.
        """
        print("\n[INFO] Creating preprocessing pipeline")
        start_time = time.time()
        
        transformers = []
        
        # Numeric features preprocessing
        if self.numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numeric', numeric_transformer, self.numeric_features))
        
        # Categorical features preprocessing
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('categorical', categorical_transformer, self.categorical_features))
        
        # Create the preprocessor
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # Full preprocessing pipeline
        preprocessing_steps = [('preprocessor', self.preprocessor)]
        
        # Apply PCA if specified
        if pca_components is not None and pca_components > 0:
            preprocessing_steps.append(('pca', PCA(n_components=pca_components)))
            print(f"[INFO] PCA will be applied with {pca_components} components")
        
        # Apply feature selection if specified
        if feature_selection is not None and feature_selection > 0:
            if self.problem_type == 'classification':
                preprocessing_steps.append(
                    ('feature_selection', SelectKBest(f_classif, k=feature_selection))
                )
            else:
                preprocessing_steps.append(
                    ('feature_selection', SelectKBest(mutual_info_regression, k=feature_selection))
                )
            print(f"[INFO] Feature selection will be applied to keep top {feature_selection} features")
        
        self.preprocessing_pipeline = Pipeline(steps=preprocessing_steps)
        
        print(f"[INFO] Preprocessing pipeline created in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def train_models(self):
        """
        Train multiple models and select the best one.
        """
        print("\n[INFO] Training multiple models")
        overall_start_time = time.time()
        
        # Define models to train based on problem type
        if self.problem_type == 'classification':
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVC': SVC(random_state=42, probability=True)
            }
            
            # Define parameter grids for each model
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'LogisticRegression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga']
                },
                'SVC': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            # For classification, we'll optimize for F1 score
            scoring = 'f1_weighted'
            
        else:  # regression
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'SVR': SVR()
            }
            
            # Define parameter grids for each model
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'LinearRegression': {},  # No hyperparameters to tune
                'Ridge': {
                    'alpha': [0.1, 1.0, 10.0]
                },
                'Lasso': {
                    'alpha': [0.1, 1.0, 10.0]
                },
                'SVR': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            # For regression, we'll optimize for negative MSE
            scoring = 'neg_mean_squared_error'
        
        # Train and evaluate models
        model_results = {}
        best_score_comparison = -np.inf if self.problem_type == 'classification' else -np.inf
        
        # Fit the preprocessing pipeline on the training data
        print("[INFO] Fitting preprocessing pipeline on training data")
        X_train_transformed = self.preprocessing_pipeline.fit_transform(self.X_train)
        X_test_transformed = self.preprocessing_pipeline.transform(self.X_test)
        
        # If target is categorical for classification problems, encode it
        if self.problem_type == 'classification' and (isinstance(self.y_train, pd.Series) and self.y_train.dtype == 'object'):
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            y_test_encoded = self.label_encoder.transform(self.y_test)
        else:
            y_train_encoded = self.y_train
            y_test_encoded = self.y_test
        
        # Keep track of model rankings
        self.model_rankings = []
        
        for model_name, model in models.items():
            print(f"\n[INFO] Training {model_name}")
            model_start_time = time.time()
            
            # Perform grid search for hyperparameter tuning
            param_grid = param_grids[model_name]
            if param_grid:
                # Adjust CV based on use_cv parameter
                cv_value = self.cv_folds if self.use_cv else 5
                
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=cv_value, 
                    scoring=scoring, 
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_transformed, y_train_encoded)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_cv_score = grid_search.best_score_
                
                # Store detailed CV results if using cross-validation
                if self.use_cv:
                    self.cv_results[model_name] = {
                        'cv_results': grid_search.cv_results_,
                        'best_params': best_params,
                        'best_cv_score': best_cv_score
                    }
                    
                print(f"[INFO] Best parameters: {best_params}")
                print(f"[INFO] Best CV score ({cv_value}-fold): {best_cv_score:.4f}")
            else:
                best_model = model
                best_model.fit(X_train_transformed, y_train_encoded)
                best_params = {}
                
                # Use the specified number of folds if using cross-validation
                cv_value = self.cv_folds if self.use_cv else 5
                best_cv_score = np.mean(cross_val_score(best_model, X_train_transformed, y_train_encoded, cv=cv_value, scoring=scoring))
                print(f"[INFO] CV score ({cv_value}-fold): {best_cv_score:.4f}")
                
            # Make predictions on test set
            y_pred = best_model.predict(X_test_transformed)
            
            # Calculate metrics based on problem type
            if self.problem_type == 'classification':
                accuracy = accuracy_score(y_test_encoded, y_pred)
                precision = precision_score(y_test_encoded, y_pred, average='weighted')
                recall = recall_score(y_test_encoded, y_pred, average='weighted')
                f1 = f1_score(y_test_encoded, y_pred, average='weighted')
                
                # Get probability predictions if the model supports it
                try:
                    y_proba = best_model.predict_proba(X_test_transformed)
                except:
                    y_proba = None
                
                test_score = f1  # Use F1 as the primary metric
                
                model_results[model_name] = {
                    'model': best_model,
                    'params': best_params,
                    'cv_score': best_cv_score,
                    'test_score': test_score,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'training_time': time.time() - model_start_time
                }
                
                print(f"[INFO] Test accuracy: {accuracy:.4f}")
                print(f"[INFO] Test precision: {precision:.4f}")
                print(f"[INFO] Test recall: {recall:.4f}")
                print(f"[INFO] Test F1 score: {f1:.4f}")
                
            else:  # regression
                mse = mean_squared_error(y_test_encoded, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_encoded, y_pred)
                
                test_score = -mse  # Negative MSE (higher is better)
                
                model_results[model_name] = {
                    'model': best_model,
                    'params': best_params,
                    'cv_score': best_cv_score,
                    'test_score': test_score,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'training_time': time.time() - model_start_time
                }
                
                print(f"[INFO] Test MSE: {mse:.4f}")
                print(f"[INFO] Test RMSE: {rmse:.4f}")
                print(f"[INFO] Test R²: {r2:.4f}")
            
            print(f"[INFO] {model_name} training completed in {time.time() - model_start_time:.2f} seconds")
            
            # Update the best model if current model is better
            if test_score > best_score_comparison:
                self.best_model = best_model
                self.best_score = test_score
                self.best_params = best_params
                self.best_model_name = model_name
                best_score_comparison = test_score
            
            # Add to model rankings
            self.model_rankings.append({
                'model_name': model_name,
                'test_score': test_score,
                'cv_score': best_cv_score,
                'training_time': model_results[model_name]['training_time']
            })
        
        # Sort model rankings
        self.model_rankings = sorted(self.model_rankings, key=lambda x: x['test_score'], reverse=True)
        
        # Extract feature importances if available
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature names from the preprocessor
            try:
                feature_names = self.get_feature_names()
                self.feature_importances = dict(zip(feature_names, self.best_model.feature_importances_))
            except:
                self.feature_importances = dict(enumerate(self.best_model.feature_importances_))
        
        # Calculate overall execution time
        self.execution_time = time.time() - overall_start_time
        print(f"\n[INFO] Model training completed in {self.execution_time:.2f} seconds")
        print(f"[INFO] Best model: {self.best_model_name} with score: {self.best_score:.4f}")
        
        return self
    
    def get_feature_names(self):
        """Get feature names from the preprocessing pipeline."""
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            return self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            feature_names = []
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'numeric':
                    feature_names.extend(features)
                elif name == 'categorical':
                    # For categorical features, get the encoded feature names
                    if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    else:
                        cat_features = [f"{feature}_{cat}" for feature in features 
                                        for cat in transformer.named_steps['onehot'].categories_[features.index(feature)]]
                    feature_names.extend(cat_features)
            return feature_names
    
    def make_predictions(self, data_slice=None, test_slice=None):
        """
        Make predictions using the best model.
        
        Parameters:
        -----------
        data_slice : pandas.DataFrame, optional
            Data to make predictions on. If None, predictions are made on the test set.
        test_slice : pandas.DataFrame, optional
            Separate test data with known target values for evaluation. Should contain
            both features and the target column.
        """
        print("\n[INFO] Making predictions")
        start_time = time.time()
        
        # Determine what data to use for predictions
        if data_slice is not None:
            data_to_predict = data_slice
            actual_values = None
            print(f"[INFO] Making predictions on provided data slice of shape {data_slice.shape}")
        elif test_slice is not None:
            # Extract features and target from test slice
            if self.target in test_slice.columns:
                data_to_predict = test_slice.drop(columns=[self.target])
                actual_values = test_slice[self.target]
                print(f"[INFO] Making predictions on provided test slice of shape {test_slice.shape}")
            else:
                raise ValueError(f"Test slice does not contain the target column '{self.target}'")
        else:
            data_to_predict = self.X_test
            actual_values = self.y_test
            print("[INFO] No data slice provided. Using test set.")
        
        # Transform the data
        X_transformed = self.preprocessing_pipeline.transform(data_to_predict)
        
        # Make predictions
        predictions = self.best_model.predict(X_transformed)
        
        # For classification, try to get probability predictions
        if self.problem_type == 'classification':
            try:
                probabilities = self.best_model.predict_proba(X_transformed)
                
                # Convert label indices back to original class names if we encoded them
                if hasattr(self, 'label_encoder'):
                    predictions = self.label_encoder.inverse_transform(predictions)
                    class_names = self.label_encoder.classes_
                    probability_df = pd.DataFrame(probabilities, columns=[f"prob_{c}" for c in class_names])
                else:
                    probability_df = pd.DataFrame(probabilities, columns=[f"prob_{c}" for c in range(probabilities.shape[1])])
                
                # Create a DataFrame with predictions and probabilities
                self.prediction_results = pd.DataFrame({
                    'prediction': predictions
                })
                self.prediction_results = pd.concat([self.prediction_results, probability_df], axis=1)
                
            except:
                # If probabilities are not available
                self.prediction_results = pd.DataFrame({
                    'prediction': predictions
                })
        else:
            # For regression
            self.prediction_results = pd.DataFrame({
                'prediction': predictions
            })
        
        # Add actual values if available
        if actual_values is not None:
            self.prediction_results['actual'] = actual_values.reset_index(drop=True)
            self.prediction_results['error'] = self.prediction_results['actual'] - self.prediction_results['prediction']
            
            # Calculate metrics
            if self.problem_type == 'classification':
                self.metrics = {
                    'accuracy': accuracy_score(actual_values, predictions),
                    'precision': precision_score(actual_values, predictions, average='weighted'),
                    'recall': recall_score(actual_values, predictions, average='weighted'),
                    'f1': f1_score(actual_values, predictions, average='weighted')
                }
            else:
                self.metrics = {
                    'mse': mean_squared_error(actual_values, predictions),
                    'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
                    'r2': r2_score(actual_values, predictions)
                }
            
            print("[INFO] Prediction metrics:", self.metrics)
        
        # Save predictions to CSV
        predictions_path = os.path.join(self.output_dir, 'predictions.csv')
        self.prediction_results.to_csv(predictions_path, index=False)
        print(f"[INFO] Predictions saved to {predictions_path}")
        
        print(f"[INFO] Predictions made in {time.time() - start_time:.2f} seconds")
        
        # Return self for method chaining
        return self
    
    def save_model(self):
        """
        Save the best model, preprocessing pipeline, and metadata.
        """
        print("\n[INFO] Saving model and results")
        start_time = time.time()
        
        # Create a full pipeline including preprocessing and the model
        full_pipeline = Pipeline([
            ('preprocessing', self.preprocessing_pipeline),
            ('model', self.best_model)
        ])
        
        # Save the full pipeline
        model_path = os.path.join(self.output_dir, 'models', 'full_pipeline.joblib')
        joblib.dump(full_pipeline, model_path)
        
        # Save feature importances if available
        if self.feature_importances is not None:
            importances_df = pd.DataFrame({
                'feature': list(self.feature_importances.keys()),
                'importance': list(self.feature_importances.values())
            }).sort_values('importance', ascending=False)
            
            importances_path = os.path.join(self.output_dir, 'reports', 'feature_importances.csv')
            importances_df.to_csv(importances_path, index=False)
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            importances_df.sort_values('importance').plot(kind='barh', x='feature', y='importance')
            plt.title('Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'feature_importances.png'))
            plt.close()
        
        # Save model rankings
        if self.model_rankings:
            rankings_df = pd.DataFrame(self.model_rankings)
            rankings_path = os.path.join(self.output_dir, 'reports', 'model_rankings.csv')
            rankings_df.to_csv(rankings_path, index=False)
            
            # Plot model comparison
            plt.figure(figsize=(10, 6))
            sns.barplot(x='test_score', y='model_name', data=rankings_df, palette='viridis')
            plt.title('Model Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'model_comparison.png'))
            plt.close()
        
        # Save metadata
        metadata = {
            'problem_type': self.problem_type,
            'best_model': self.best_model_name,
            'best_model_params': self.best_params,
            'best_model_score': float(self.best_score),
            'execution_time': float(self.execution_time),         
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        
        metadata_path = os.path.join(self.output_dir, 'reports', 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        # Generate final report
        self.generate_report()
        
        print(f"[INFO] Model saved to {model_path}")
        print(f"[INFO] Save operation completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def generate_report(self):
        """
        Generate a comprehensive HTML report.
        """
        report_path = os.path.join(self.output_dir, 'reports', 'final_report.html')
        
        # Create a basic HTML report
        with open(report_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AutoML Pipeline Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ display: inline-block; width: 200px; margin: 10px; padding: 15px; text-align: center; background-color: #f8f9fa; border-radius: 5px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .metric-label {{ font-size: 14px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>AutoML Pipeline Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <p>Dataset size: {self.data.shape[0]} rows, {self.data.shape[1]} columns</p>
                    <p>Problem type: {self.problem_type.capitalize()}</p>
                    <p>Target column: {self.target}</p>
                    <p>Number of numeric features: {len(self.numeric_features)}</p>
                    <p>Number of categorical features: {len(self.categorical_features)}</p>
                </div>
                
                <div class="section">
                    <h2>Best Model</h2>
                    <p>Best model algorithm: {self.best_model_name}</p>
                    <p>Best model parameters: {self.best_params}</p>
                    <p>Model training time: {self.execution_time:.2f} seconds</p>
                </div>
            """)
            
            # Add metrics section based on problem type
            if self.metrics:
                f.write('<div class="section"><h2>Model Performance</h2>')
                for metric_name, metric_value in self.metrics.items():
                    f.write(f"""
                    <div class="metric">
                        <div class="metric-label">{metric_name.upper()}</div>
                        <div class="metric-value">{metric_value:.4f}</div>
                    </div>
                    """)
                f.write('</div>')
            
            # Add model comparison table
            if self.model_rankings:
                f.write("""
                <div class="section">
                    <h2>Model Comparison</h2>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Test Score</th>
                            <th>CV Score</th>
                            <th>Training Time (s)</th>
                        </tr>
                """)
                
                for i, model in enumerate(self.model_rankings):
                    f.write(f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{model['model_name']}</td>
                        <td>{model['test_score']:.4f}</td>
                        <td>{model['cv_score']:.4f}</td>
                        <td>{model['training_time']:.2f}</td>
                    </tr>
                    """)
                
                f.write('</table></div>')
            
            # Add feature importances if available
            if self.feature_importances:
                f.write("""
                <div class="section">
                    <h2>Top Feature Importances</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                        </tr>
                """)
                
                # Sort feature importances and show top 20
                sorted_importances = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
                for feature, importance in sorted_importances:
                    f.write(f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                    """)
                
                f.write('</table></div>')
            
            # Close the HTML document
            f.write("""
                <div class="section">
                    <h2>Files Generated</h2>
                    <ul>
                        <li>Full model pipeline: <code>models/full_pipeline.joblib</code></li>
                        <li>Predictions: <code>predictions.csv</code></li>
                        <li>Feature importances: <code>reports/feature_importances.csv</code></li>
                        <li>Model rankings: <code>reports/model_rankings.csv</code></li>
                        <li>Metadata: <code>reports/metadata.json</code></li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Usage Instructions</h2>
                    <p>To load and use the saved model:</p>
                    <pre>
                    import joblib
                    model = joblib.load('models/full_pipeline.joblib')
                    predictions = model.predict(new_data)
                    </pre>
                </div>
            </body>
            </html>
            """)
        
        print(f"[INFO] Report generated: {report_path}")
        return self
    
    def run_pipeline(self, data_path, target_column, test_size=0.2, random_state=42, 
                cv_folds=5, use_cv=False, pca_components=None, feature_selection=None, 
                id_columns=None, drop_columns=None, test_slice=None, prediction_slice=None):
        """
        Run the complete AutoML pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the data file
        target_column : str
            Name of the target column
        test_size : float, optional (default=0.2)
            Proportion of data to use for testing
        random_state : int, optional (default=42)
            Random seed for reproducibility
        cv_folds : int, optional (default=5)
            Number of folds to use in cross-validation
        use_cv : bool, optional (default=False)
            Whether to use cross-validation instead of a simple train-test split
        pca_components : int, optional
            Number of PCA components to use
        feature_selection : int, optional
            Number of features to select
        id_columns : list, optional
            List of ID columns to exclude
        drop_columns : list, optional
            Additional columns to drop
        test_slice : pandas.DataFrame, optional
            External test data with known target values
        prediction_slice : pandas.DataFrame, optional
            Data to make predictions on without known target values
        """
        print("\n" + "="*80)
        print("Starting AutoML Pipeline")
        print("="*80)
        
        start_time = time.time()
        
        # Run the pipeline steps
        self.ingest_data(data_path, target_column, id_columns, drop_columns)
        self.exploratory_data_analysis()
        self.prepare_data(test_size, random_state, cv_folds, use_cv)
        self.create_preprocessing_pipeline(pca_components, feature_selection)
        self.train_models()
        
        # Make predictions on specified data or default test set
        self.make_predictions(data_slice=prediction_slice, test_slice=test_slice)
        self.save_model()
        
        # Calculate and display total execution time
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        print("\n" + "="*80)
        print(f"AutoML Pipeline Completed in {int(minutes)} minutes and {seconds:.2f} seconds")
        print(f"Best model: {self.best_model_name}")
        if self.metrics:
            print("Performance metrics:", self.metrics)
        print(f"All outputs saved to {self.output_dir}")
        print("="*80)
        
        return self

# Streamlit App
def main():
    st.title("AutoML Pipeline with Streamlit")
    st.write("Upload your dataset, select the target column, and train models automatically.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)
        
        st.write("### Dataset Preview")
        st.write(data.head())
        
        # Select target column
        target_column = st.selectbox("Select the target column", data.columns)
        
        # Select ID columns to drop
        id_columns = st.multiselect("Select ID columns to drop (optional)", data.columns)
        
        # Select additional columns to drop
        drop_columns = st.multiselect("Select additional columns to drop (optional)", data.columns)
        
        # Train-test split parameters
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random state", value=42)
        
        # Cross-validation options
        use_cv = st.checkbox("Use cross-validation for training")
        cv_folds = st.number_input("Number of CV folds", value=5, min_value=2, max_value=10) if use_cv else 5
        
        # PCA and feature selection
        pca_components = st.number_input("Number of PCA components (optional)", min_value=0, value=0)
        feature_selection = st.number_input("Number of features to select (optional)", min_value=0, value=0)
        
        # Train button
        if st.button("Train Models"):
            st.write("### Training Models...")
            
            # Initialize AutoMLPipeline
            automl = AutoMLPipeline(output_dir='streamlit_output')
            
            # Run the pipeline
            automl.run_pipeline(
                data_path=uploaded_file.name,
                target_column=target_column,
                id_columns=id_columns,
                drop_columns=drop_columns,
                test_size=test_size,
                random_state=random_state,
                cv_folds=cv_folds,
                use_cv=use_cv,
                pca_components=pca_components if pca_components > 0 else None,
                feature_selection=feature_selection if feature_selection > 0 else None
            )
            
            # Display results
            st.write("### Best Model")
            st.write(f"Algorithm: {automl.best_model_name}")
            st.write(f"Parameters: {automl.best_params}")
            st.write(f"Score: {automl.best_score:.4f}")
            
            st.write("### Model Metrics")
            st.write(automl.metrics)
            
            st.write("### Model Rankings")
            st.write(pd.DataFrame(automl.model_rankings))
            
            st.write("### Feature Importances")
            if automl.feature_importances:
                st.write(pd.DataFrame({
                    'Feature': list(automl.feature_importances.keys()),
                    'Importance': list(automl.feature_importances.values())
                }).sort_values('Importance', ascending=False))
            
            st.write("### Predictions")
            st.write(automl.prediction_results)
            
            st.write("### Download Results")
            st.write("All outputs are saved in the `streamlit_output` directory.")

if __name__ == "__main__":
    main()