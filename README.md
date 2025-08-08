<xaiArtifact artifact_id="0e7da114-bce2-494b-b1e4-3c474367b4c5" artifact_version_id="c5e23b28-7a6c-41f7-8689-b84e820f13e3" title="README.md" contentType="text/markdown">

# ML-Powered Job Application Success Predictor

## Overview
This project develops a machine learning model to predict the success of job applications based on resume data. Using a dataset from Kaggle, it classifies resumes into job categories by processing and analyzing their textual content. The project employs multiple machine learning models, including Logistic Regression, Random Forest, and XGBoost, to achieve accurate predictions. The cleaned and processed text data is vectorized using TF-IDF, and the models are evaluated based on their classification performance.

## Features
- **Dataset**: Utilizes the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) from Kaggle.
- **Data Preprocessing**: Cleans resume text by removing URLs, special characters, and unnecessary spaces.
- **Exploratory Data Analysis (EDA)**: Visualizes category distribution and text length distribution using Seaborn and Plotly.
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Text Vectorization**: Employs TF-IDF for transforming text data into numerical features.
- **Model Evaluation**: Uses classification reports and confusion matrices to assess model performance.
- **Model Persistence**: Saves the trained XGBoost model for future use.

## Requirements
To run this project, ensure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- xgboost
- tensorflow
- joblib

You can install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost tensorflow joblib
```

Additionally, you need a Kaggle API key (`kaggle.json`) to download the dataset.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ml-job-application-predictor.git
   cd ml-job-application-predictor
   ```

2. **Set Up Kaggle API**:
   - Obtain your `kaggle.json` file from Kaggle.
   - Place it in the project directory or upload it when prompted in Google Colab.
   - Run the following commands to set up the Kaggle API:
     ```bash
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the Dataset**:
   ```bash
   kaggle datasets download -d snehaanbhawal/resume-dataset
   unzip resume-dataset.zip -d resume_dataset
   ```

## Usage
1. **Run the Jupyter Notebook or Python Script**:
   - Open the provided notebook (`ml_job_predictor.ipynb`) in Jupyter or Google Colab, or run the equivalent Python script.
   - Ensure the dataset is available in the `resume_dataset/Resume` directory.

2. **Key Steps in the Code**:
   - **Load and Clean Data**: Loads the resume dataset and applies text cleaning (removing URLs, special characters, etc.).
   - **EDA**: Visualizes category and text length distributions.
   - **Train-Test Split**: Splits data into training (80%) and testing (20%) sets.
   - **Vectorization**: Converts text data into TF-IDF features.
   - **Model Training**: Trains Logistic Regression, Random Forest, and XGBoost models.
   - **Evaluation**: Displays classification reports and confusion matrices for each model.
   - **Model Saving**: Saves the XGBoost model as `xgb_model.joblib`.

3. **Example Command**:
   ```bash
   jupyter notebook ml_job_predictor.ipynb
   ```

## Project Structure
```
ml-job-application-predictor/
│
├── resume_dataset/Resume/Resume.csv   # Dataset
├── xgb_model.joblib                   # Saved XGBoost model
├── ml_job_predictor.ipynb             # Jupyter Notebook with the code
├── README.md                          # This file
```

## Results
- **Category Distribution**: The dataset contains resumes across multiple job categories, visualized using a bar plot and pie chart.
- **Text Length Analysis**: Most resumes have a word count between 100 and 1000 words, as shown in the histogram.
- **Model Performance**:
  - Logistic Regression: Provides baseline performance with good accuracy.
  - Random Forest: Captures complex patterns but may overfit.
  - XGBoost: Achieves high accuracy with robust handling of imbalanced classes.
  - Detailed classification reports and confusion matrices are generated for each model.

## Future Improvements
- Incorporate deep learning models like LSTM or BERT for better text understanding.
- Add feature engineering to include resume metadata (e.g., years of experience).
- Implement cross-validation for more robust model evaluation.
- Deploy the model as a web application for real-time predictions.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [Snehaan Bhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
- Built with Python, Scikit-learn, XGBoost, and TensorFlow.

</xaiArtifact>
