# BERTImpactGenome  
**DSC 672 - Group 5 - Impact Genome**  
- Julia Aptekar, DePaul University, japtekar@depaul.edu  
- John Leniart, DePaul University, jleniart@depaul.edu  
- Arham Mehdi, DePaul University, kmehdi@depaul.edu  
- Natalie Olechno, DePaul University, nolechno@depaul.edu  

---

## Project Overview

This project aims to automate the classification of nonprofit program outcomes using a hierarchical BERT-based NLP model. Built as part of the DSC 672 Data Science Capstone at DePaul University, the model significantly outperformed the existing solution developed with Google Vertex AI.

Impact Genome maintains a registry of social program outcomes. These outcomes were previously categorized manually or with low-accuracy models. Our team developed a custom NLP pipeline to predict outcomes across three hierarchical levels:

1. **Impact Area** – Broad category (e.g., Education, Health)  
2. **Genome** – Sub-category  
3. **Outcome ID** – Specific program result  

**Our BERT model achieved:**
- 87.3% accuracy for Impact Area  
- 86.3% accuracy for Genome  
- 66.3% accuracy for Outcome ID  
(*Compared to 52% baseline from Google Vertex AI*)



## 📊 Exploratory Data Analysis (`DSC672_Group5_EDA.ipynb`)
- Can be run locally, on Google Colab, or Kaggle  
- Upload `Validated Data from Heather.xlsx` and `Combined Data.xlsx` to your working folder  
- Adjust paths in the code to match your environment  

**Libraries Used:**
- `pandas` – data loading, manipulation, preprocessing  
- `numpy` – numerical operations  
- `re` – regular expressions  
- `matplotlib.pyplot`, `seaborn`, `wordcloud` – visualizations  



## 🧪 Logistic Regression (`DSC672_Group5_LogisticRegression.ipynb`)
- Can be run locally, on Google Colab, or Kaggle  
- Upload both Excel files to your working folder  
- Adjust file paths as needed  

**Libraries Used:**
- `pandas`, `numpy`  
- `sklearn.model_selection` – `train_test_split`, `GridSearchCV`  
- `sklearn.linear_model` – `LogisticRegression`  
- `sklearn.metrics` – `accuracy_score`, `f1_score`, `precision_score`, `recall_score`  
- `sklearn.preprocessing` – `OneHotEncoder`  
- `sklearn.compose` – `ColumnTransformer`  
- `sklearn.pipeline` – `Pipeline`  



## 🔰 Baseline BERT Model (`BaselineBERT.ipynb`)
- Developed in Kaggle, also compatible with Google Colab  
- Upload `Combined Data.xlsx`  
- Make sure to adjust paths and model save location  

**GPU :**
- Kaggle: GPU P100 or T4X2  
- Colab: GPU T4X2  



## BERT Model 1 (`Model_1.ipynb`)
- Developed using Kaggle; compatible with any GPU environment  
- Upload `Combined Data.xlsx`  
- Adjust paths in code accordingly  

**GPU Requirements:**
- Kaggle: GPU P100 or T4X2 (falls back to CPU if unavailable)  
- Colab: GPU T4X2  

**Libraries Used:**
- `pandas`, `numpy`, `re`  
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`  
- `transformers`  
- `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.utils.class_weight`, `sklearn.metrics`  



## BERT Model 2 (`Model_2_and_Pipeline.ipynb`)
- Developed using Kaggle; also runs in Google Colab  
- Upload `Combined Data.xlsx` and adjust file paths  

**GPU Requirements:**
- Kaggle: GPU P100 or T4X2  
- Colab: GPU T4X2  



## BERT Model 3 (`Model_3_and_Pipeline.ipynb`)
- Developed using Jupyter Notebook; compatible with any GPU environment  
- Upload `Combined Data.xlsx` and adjust file paths  

**Libraries Used:**
- `os`, `time`, `re`  
- `pandas`, `numpy`  
- `matplotlib.pyplot`, `seaborn`  
- `transformers`  
- `sklearn.model_selection`, `sklearn.metrics`, `sklearn.preprocessing`  
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `torch.amp`, `torch.serialization`  
- `tqdm`  



## 📦 Trained Models
- `best_hierarchical_bert_model2.pt` → Model 2  
- `basic_bert_model.pt` → Baseline BERT  
- `hierarchical_bert_model3.pt` → Model 3  



##  Instructions
1. Upload required datasets (`Combined Data.xlsx`, `Validated Data from Heather.xlsx`)  
2. Adjust paths in all notebooks to your working environment  
3. Ensure GPU runtime (Kaggle or Colab) is properly configured  
4. Run cells step by step as indicated in each notebook  

 
