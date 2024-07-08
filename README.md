# AutoInsights

AutoInsights is an **AutoML** platform designed to facilitate classification and regression tasks. It simplifies and automates the process of machine learning model development, allowing users to focus on deriving insights from their data without getting bogged down by technical complexities.

## Features and Functionality

### Key Features
- **Task Selection**: Allows users to choose between regression and classification tasks.
- **Data Upload**: Users can upload their datasets in CSV format.
- **Data Profiling**: Provides a comprehensive data profiling report using ydata-profiling, helping users understand their data better.
- **Model Training and Evaluation**: Enables users to train and evaluate models with various metrics, tailored for both regression and classification tasks.
- **Downloadable Models**: After training, users can download the best performing model for deployment or further analysis.

### Development Tools
- **Streamlit**: Used for building the front-end interface, providing a responsive and interactive user experience.
- **PyCaret**: A low-code machine learning library in Python that automates the entire machine learning workflow, from data preprocessing to model deployment.
- **ydata-profiling**: Employed for generating detailed data profiling reports.

## Problem Statement
Many data scientists and analysts face challenges in managing the various tasks involved in creating, training, and evaluating machine learning models, particularly for regression and classification tasks. These challenges include data profiling, selecting appropriate evaluation metrics, encoding target variables, and managing model files. AutoInsights aims to address these challenges by providing an automated and user-friendly platform that streamlines the entire workflow, enabling users to derive insights more efficiently and effectively.

## Getting Started
To get started with AutoInsights, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/phyosandarwin/auto-insights.git
   
2. Navigate to project directory:
    ```bash
    cd auto-insights

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

4. Run the Streamlit app:
    ```bash
    streamlit run app.py

# View Demo Video
[Video Link](https://youtu.be/TXbRi4xD70Y)
