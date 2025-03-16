import streamlit as st
import pandas as pd
from automl_pipeline import AutoMLPipeline  # Import the AutoMLPipeline class
import os
import zipfile

def create_zip(output_dir, zip_name):
    """
    Create a zip file from the contents of the output directory.
    """
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=output_dir)
                zipf.write(file_path, arcname)

def main():
    st.title("AutoML Pipeline with Streamlit")
    st.write("Upload your training dataset, select the target column, and train models automatically.")
    
    # Initialize session state variables
    if 'automl' not in st.session_state:
        st.session_state.automl = None
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'id_columns' not in st.session_state:
        st.session_state.id_columns = []
    if 'drop_columns' not in st.session_state:
        st.session_state.drop_columns = []
    if 'train_file_name' not in st.session_state:
        st.session_state.train_file_name = None
    
    # File upload for training set
    st.write("### Upload Training Set")
    train_file = st.file_uploader("Upload your training dataset (CSV or Excel)", type=["csv", "xlsx"], key="train_file")
    
    if train_file is not None:
        # Load training data
        if st.session_state.train_data is None or train_file.name != st.session_state.train_file_name:
            if train_file.name.endswith('.csv'):
                st.session_state.train_data = pd.read_csv(train_file)
            elif train_file.name.endswith(('.xls', '.xlsx')):
                st.session_state.train_data = pd.read_excel(train_file)
            st.session_state.train_file_name = train_file.name
        
        st.write("#### Training Set Preview")
        st.write(st.session_state.train_data.head())
        
        # Select target column
        target_column = st.selectbox("Select the target column", st.session_state.train_data.columns)
        
        # Select ID columns to drop
        id_columns = st.multiselect("Select ID columns to drop (optional)", st.session_state.train_data.columns)
        
        # Select additional columns to drop
        drop_columns = st.multiselect("Select additional columns to drop (optional)", st.session_state.train_data.columns)
        
        # Check if target column, ID columns, or drop columns have changed
        if (st.session_state.target_column != target_column or
            st.session_state.id_columns != id_columns or
            st.session_state.drop_columns != drop_columns):
            
            # Update session state
            st.session_state.target_column = target_column
            st.session_state.id_columns = id_columns
            st.session_state.drop_columns = drop_columns
            
            # Re-run EDA
            st.session_state.automl = AutoMLPipeline(output_dir='streamlit_output')
            st.session_state.automl.ingest_data(train_file.name, target_column, id_columns, drop_columns)
            st.session_state.automl.exploratory_data_analysis()
        
        # Perform EDA on the training set
        st.write("### Exploratory Data Analysis (EDA)")
        
        # Display EDA plots with trendlines
        st.write("#### Numeric Feature Distributions with Trendlines")
        if st.session_state.automl is not None and st.session_state.automl.numeric_features:
            for feature in st.session_state.automl.numeric_features:
                st.image(os.path.join('streamlit_output', 'plots', f'{feature}_distribution_trendline.png'))
        
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
            
            # Run the pipeline on the training set
            st.session_state.automl.run_pipeline(
                data_path=train_file.name,
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
            st.write(f"Algorithm: {st.session_state.automl.best_model_name}")
            st.write(f"Parameters: {st.session_state.automl.best_params}")
            st.write(f"Score: {st.session_state.automl.best_score:.4f}")
            
            st.write("### Model Metrics")
            st.write(st.session_state.automl.metrics)
            
            st.write("### Model Rankings")
            st.write(pd.DataFrame(st.session_state.automl.model_rankings))
            
            st.write("### Feature Importances")
            if st.session_state.automl.feature_importances:
                st.write(pd.DataFrame({
                    'Feature': list(st.session_state.automl.feature_importances.keys()),
                    'Importance': list(st.session_state.automl.feature_importances.values())
                }).sort_values('Importance', ascending=False))
            
            # Create a zip file of the output directory
            zip_name = "streamlit_output.zip"
            create_zip('streamlit_output', zip_name)
            
            # Provide a download button for the zip file
            with open(zip_name, "rb") as f:
                st.download_button(
                    label="Download Results as Zip. Download before using prediction set.",
                    data=f,
                    file_name=zip_name,
                    mime="application/zip"
                )
    
    # Section for making predictions on a prediction set
    if st.session_state.automl is not None and st.session_state.automl.best_model is not None:
        st.write("### Make Predictions on a New Dataset")
        st.write("Upload a prediction set to make predictions using the trained model.")
        
        # File upload for prediction set
        prediction_file = st.file_uploader("Upload your prediction dataset (CSV or Excel)", type=["csv", "xlsx"], key="prediction_file")
        
        if prediction_file is not None:
            # Load prediction data
            if st.session_state.prediction_data is None or prediction_file.name != st.session_state.prediction_file_name:
                if prediction_file.name.endswith('.csv'):
                    st.session_state.prediction_data = pd.read_csv(prediction_file)
                elif prediction_file.name.endswith(('.xls', '.xlsx')):
                    st.session_state.prediction_data = pd.read_excel(prediction_file)
                st.session_state.prediction_file_name = prediction_file.name
            
            st.write("#### Prediction Set Preview")
            st.write(st.session_state.prediction_data.head())
            
            # Make predictions on the prediction set
            st.session_state.automl.make_predictions(data_slice=st.session_state.prediction_data)
            
            st.write("### Predictions on Prediction Set")
            st.write(st.session_state.automl.prediction_results)
            
            # Provide a download button for the predictions
            predictions_csv = st.session_state.automl.prediction_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=predictions_csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()