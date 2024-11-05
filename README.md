# DTSE Data Engineer (ETL) Assignment

## Overview
The `process.py` script contains functionality for data preprocessing and making predictions using a provided model. To execute the script, run:

```bash
python process.py
```

Upon execution, the script generates a database file, `housing_predictions.db`, which can be viewed with tools like **DB Browser for SQLite**. Additionally, example outputs are printed to showcase the script's functionality.

## Data Processing Steps
The data undergoes the following transformations:

1. **Exclude redundant columns**
2. **Rename columns** to match the trained model's attribute names
3. **Divide data** into numeric and non-numeric columns
4. **Replace NULL data with NaN** in both arrays
5. **Impute missing values** using `KNNImputer` and/or `SimpleImputer`
6. **Scale numeric data** with `StandardScaler` in the numeric array
7. **Encode non-numeric data** using `OneHotEncoder` to create binary (1s and 0s) features
8. **Concatenate both arrays** back into a single dataset

The final data array is converted into a DataFrame to ensure compatibility with the provided model.

## Database Integration
The preprocessed data and prediction results are transformed into DataFrames and then inserted into a SQLite3 database (a library included with Python's standard library).

## Additional Files

- **`alternativeSQL.py`**: An alternative script for creating tables in the database.
- **`assignment.txt`**: Contains the text from the original `README.md` file.
- **`optional.txt`**: Includes a discussion on topics covered in the assignment tasks.

## Note
The `requirements.txt` file remains unchanged.