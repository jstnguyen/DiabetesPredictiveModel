
# Diabetes Predictive Model for ECS 171 - Machine Learning Fall 2023
Our project focuses on developing a predictive model to diagnose diabetes using machine learning techniques. We aim to enhance the accuracy of diabetes diagnosis by utilizing patients' demographic information and medical history.

## Dataset
Our model is trained on the "Diabetes Prediction Dataset" from Kaggle: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). This dataset comprises essential medical and demographic data for accurate diabetes prediction.

## Installation Instructions

**Python and Libraries:**
1. Ensure Python is installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).
2. (Optional but Recommended) Create a virtual environment:
   ```bash
   python -m venv myenv
   # Windows: myenv\Scripts\activate
   # Unix/MacOS: source myenv/bin/activate
   ```
3. Install required Python packages with [pip install](https://docs.python.org/3/installing/index.html):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn flask pickle
   ```
4. Install Jupyter Notebook (if not already installed):
   ```bash
   pip install notebook
   ```

**Flask Dependencies:**
- Install additional Flask dependencies as required by `webpg.ipynb`.

## Usage

1. **Exploratory Data Analysis (EDA):**
   - Open `diabetes_prediction_EDA.ipynb` for data visualization and initial dataset analysis.

2. **Model Building and Evaluation:**
   - Refer to `Log_Reg_Model.ipynb`, `Naive_Bayes_Model.ipynb`, and `Random_Forest_Model.ipynb` for model-specific workflows.
   - These notebooks guide through data loading, preprocessing, model training, and evaluation.
   - Be sure to run `Log_Reg_Model.ipynb` in order to create `model.pkl` and `scaler.pkl`.

3. **Web Application:**
     - The project utilizes three key files for web application:
       
        - `model.pkl`: Contains the serialized logistic regression model used by our HTML-based webpage.
        - `scaler.pkl`: Stores the serialized MinMaxScaler for normalizing input features for the model.
        - `webpg.ipynb`: Contains Flask code for a web interface, demonstrating the application of the model in a web environment.

    - To use the model, run `webpg.ipynb` and click the localhost link generated in the output terminal. Enter medical data on the webpage using values from the dataset. The model then provides a prediction regarding the presence of diabetes.

5. **General Workflow:**
   - Start with the EDA notebook, then move to the model notebooks. For an interactive web application, run the `webpg.ipynb` notebook for web deployment.

## Features
- Detailed data preprocessing and analysis to enhance prediction accuracy.
- Implements logistic regression, Naïve Bayes, and random forest models.
- Flask-based web interface for user interaction.
- User-friendly Jupyter Notebooks

## Authors 
Justin Nguyen, Daniel Heredia, Ronit Amar Bhatia, Amanda Tu, Katie Sharp
