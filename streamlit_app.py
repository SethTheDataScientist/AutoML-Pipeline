
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
            
            # Create a zip file of the output directory
            zip_name = "streamlit_output.zip"
            create_zip('streamlit_output', zip_name)
            
            # Provide a download button for the zip file
            with open(zip_name, "rb") as f:
                st.download_button(
                    label="Download Results as Zip",
                    data=f,
                    file_name=zip_name,
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()