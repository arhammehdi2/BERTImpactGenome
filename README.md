# BERTImpactGenome
DSC 672 - Group 5 - Impact Genome
Julia Aptekar, DePaul University, japtekar@depaul.edu
John Leniart, DePaul University, jleniart@depaul.edu
Arham Mehdi, DePaul University kmehdi@depaul.edu
Natalie Olechno, DePaul University, nolechno@depaul.edu

README
This project aims to automate the classification of nonprofit program outcomes using a hierarchical BERT-based NLP model. Built as part of the DSC 672 Data Science Capstone at DePaul University, the model significantly outperformed the existing solution developed with Google Vertex AI.

## Project Overview

Impact Genome maintains a registry of social program outcomes. These outcomes were previously categorized manually or with low-accuracy models. Our team developed a custom NLP pipeline to predict outcomes across three hierarchical levels:

1. **Impact Area** – Broad category (e.g., Education, Health)
2. **Genome** – Sub-category
3. **Outcome ID** – Specific program result

Our BERT model achieved:
- **87.3% accuracy** for Impact Area
- **86.3% accuracy** for Genome
- **66.3% accuracy** for Outcome ID  
(Compared to **52% baseline** from Google Vertex AI)

Exploratory Data Analysis (DSC672_Group5_EDA.ipynb)
All code can be run locally or in Google Colab or Kaggle
●	Upload the ‘Validated Data from Heather.xlsx’ and ‘Combined Data.xlsx’ files to your working folder.
●	Adjust paths in code to connect to your working folder.
●	Libraries Used:
○	pandas → data loading, manipulation, and preprocessing
○	numpy → numerical operations
○	re → regular expressions used for data cleaning
○	matplotlib.pyplot → data visualization
○	seaborn → data visualization
○	wordcloud → data visualization

Logistic Regression (DSC672_Group5_LogisticRegression.ipynb)
All code can be run locally or in Google Colab or Kaggle
●	Upload the ‘Validated Data from Heather.xlsx’ and ‘Combined Data.xlsx’ files to your working folder.
●	Adjust paths in code to connect to your working folder.
●	Libraries Used:
○	pandas -Data loading & preprocessing
○	numpy - Numerical operations
○	sklearn.model_selection → train_test_split, GridSearchCV
○	sklearn.linear_model → LogisticRegression
○	sklearn.metrics → accuracy_score, f1_score, precision_score, recall_score
○	sklearn.preprocessing → OneHotEncoder
○	sklearn.compose → ColumnTransformer
○	sklearn.pipeline → Pipeline

Baseline BERT Model (BaselineBERT.ipynb)
Model was developed using Kaggle and may also be run in Google Colab
●	Upload ‘Combined Data.xlsx’ to your working folder.
●	Adjust paths in code to connect to your working folder. Pay attention to ‘Combined Data.xlsx’ and where ‘.pt’ will be saved. 
1.	With Kaggle:
a.	Connect to GPU P100 or GPU T4X2. The code will not run otherwise. 
2.	With Google Colab:
a.	Connect to GPU T4X2

BERT Model 1 (Model 1.ipynb)
Model was developed using Kaggle and may be run in any GPU environment.
●	Upload ‘Combined Data.xlsx’ to your file directory.
●	Adjust paths in code to connect to your working folder. 
●	With Kaggle:
b.	Connect to GPU P100 or GPU T4X2. Forces CPU to run if GPU is not available.. 
3.	With Google Colab:
a.	Connect to GPU T4X2
●	Libraries Used:
○	pandas -Data loading & preprocessing
○	numpy - Numerical operations
○	torch - PyTorch deep learning framework
○	torch.nn - Neural network layers and loss functions
○	torch.optim - Optimization algorithms
○	torch.utils.data - Data handling utilities (Dataset, DataLoader)
○	transformers - Hugging Face library for BERT-based models
○	sklearn.preprocessing - Label encoding
○	sklearn.model_selection - Train-test split
○	sklearn.utils.class_weight - Compute class weights
○	sklearn.metrics- Model evaluation metrics
○	re - Regular expressions for text cleaning

BERT Model 2 (Model_2_and_Pipeline.ipynb)
Model was developed using Kaggle and may also be run in Google Colab
●	Upload ‘Combined Data.xlsx’ to your working folder.
●	Adjust paths in code to connect to your working folder. Pay attention to ‘Combined Data.xlsx’ and where ‘.pt’ will be saved. 
4.	With Kaggle:
a.	Connect to GPU P100 or GPU T4X2. The code will not run otherwise. 
5.	With Google Colab:
a.	Connect to GPU T4X2

BERT Model 3 (Model_3_and_Pipeline.ipynb)
Model was developed using Jupyter and may be run in any GPU environment
●	Upload ‘Combined Data.xlsx’ to your working folder.
●	Adjust paths in code to connect to your working folder. Pay attention to ‘Combined Data.xlsx’ and where ‘.pt’ will be saved. 
●	Libraries Used:
○	os - File management
○	time - Time tracking helper function
○	re - Data cleaning
○	pandas - Data loading & preprocessing
○	numpy - Numeric operations
○	matplotlib.pyplot - Data visualization
○	seaborn - Customized data visualization
○	transformers - Hugging Face library for BERT-based models
○	sklearn.preprocessing - Label encoding
○	sklearn.model_selection - Train-test split
○	sklearn.metrics - Evaluation metrics (accuracy, precision, recall, f1)
○	torch - PyTorch deep learning framework
○	torch.nn - Neural network layers and loss functions
○	torch.optim - Optimization algorithms
○	torch.utils.data - Data handling utilities (Dataset, DataLoader)
○	torch.amp - Automatic mixed precision scaling
○	torch.serialization - Model loading exception handling
○	tqdm - Progress visualizer

Models:
best_hierarchical_bert_model2.pt
●	Corresponds to Model_2_and_Pipeline.ipynb
basic_bert_model.pt
●	Corresponds to BaselineBERT.ipynb
hierarchical_bert_model3.pt
●	Corresponds to Model_3_and_Pipeline.ipynb
