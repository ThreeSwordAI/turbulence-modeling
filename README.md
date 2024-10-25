# Turbulence Modeling Using Machine Learning

This project applies machine learning to predict turbulence fields from RANS simulations using DNS/LES reference data. The model leverages AutoML with AutoGluon to streamline training and hyperparameter tuning.


## Dataset

This project uses the "Turbulence Modeling using Machine Learning" dataset, available on Kaggle: [Turbulence Modeling Dataset](https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset). Download the data and place it in the `data/` directory before running the code.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ThreeSwordAI/turbulence-modeling.git
   cd turbulence-modeling

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt

3. Running the Project
   ```bash
   python src/main.py





## Project Structure
```plaintext
turbulence-modeling/
│
├── data/                 # Dataset Folder
├── src/                  # Source code
│   └── main.py           # Main script to load data, train model and evaluate
├── README.md             # Project overview and Setup Instructions
├── .gitignore            # Git ignore file
└── requirements.txt      # Python dependencies

