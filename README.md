# DATA-PIPELINE-DEVELOPMENT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: J.TANUJ

*INTERN ID*: CT06DL1326

*DOMAIN*: DATA SCIENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTHOSH

## DESCRIPTION
Task 1: Data Preprocessing Pipeline using Pandas and Scikit-Learn
This task focuses on building a complete ETL (Extract, Transform, Load) pipeline using the Iris dataset. The objective is to automate the data preprocessing steps commonly required in machine learning workflows. Python libraries such as Pandas, NumPy, and Scikit-learn are used throughout the script.

Step 1: Data Loading
The script begins by loading the Iris dataset from Scikit-learn. The dataset contains measurements of iris flowers—sepal length, sepal width, petal length, and petal width—along with their species (Setosa, Versicolor, Virginica). It is converted into a Pandas DataFrame for easier handling and manipulation.

Step 2: Introducing Missing Values
To simulate a real-world scenario, a few missing values are manually introduced into the dataset. This step helps demonstrate how to handle incomplete data during preprocessing.

Step 3: Handling Missing Data
Missing numeric values are filled using the mean of their respective columns. This is a common imputation strategy that helps preserve the dataset's overall distribution.

Step 4: Label Encoding
Although the Iris dataset’s target values are already numerical, a LabelEncoder is used for demonstration purposes. This step ensures that categorical labels are converted into machine-readable numerical format.

Step 5: Feature Scaling
The feature columns are standardized using StandardScaler from Scikit-learn. Standardization ensures that each feature has zero mean and unit variance, which is important for many machine learning algorithms to perform well.

Step 6: Train/Test Split
The processed data is split into training (80%) and testing (20%) sets using train_test_split. This allows for separate model training and evaluation later in the pipeline.

Step 7: Saving Processed Data
The final preprocessed training and testing sets are saved as CSV files:

processed_train_data.csv

processed_test_data.csv
These files can be used directly for training and validating machine learning models.

Outcome
This script demonstrates a fully functional data preprocessing pipeline, automating the essential steps from raw data to clean, ready-to-use inputs. It is efficient, well-documented, and aligned with real-world data science practices. 

##OUTPUT:
![Image](https://github.com/user-attachments/assets/667ce2bc-406a-40fb-ad09-e3e1e1074afd)



