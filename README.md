# DTSE Data Engineer (ETL) Assignment

## Overview
The `process.py` script contains functionality for data preprocessing and making predictions using a provided model. To execute the script, run the following command in the terminal:

```bash
python process.py
```

Upon execution, the script generates a database file, `housing_predictions.db`, which can be viewed with **DB Browser for SQLite**. Additionally, during execution, example outputs are printed to showcase the script's functionality as per mandatory assignment point 2.

## Data Processing Steps
The data undergoes the following transformations before fed into a model:

1. **Exclude redundant columns**
2. **Rename columns** to match the trained model's attribute names
3. **Divide data** into numeric and non-numeric columns
4. **Replace NULL data with NaN** in both arrays
5. **Impute missing values** using `KNNImputer` and/or `SimpleImputer`
6. **Scale numeric data** with `StandardScaler` in the numeric array
7. **Encode non-numeric data** using `OneHotEncoder` to create binary (1s and 0s) features
8. **Concatenate both arrays** back into a single data array

The final data array is converted into a DataFrame type to ensure compatibility with the provided model.

## Database Integration
The DataFrame array is inserted into a SQLite3 database (a library included with Python's standard library). The prediction results are also transformed into DataFrames and then inserted into a SQLite3 database.

## Additional Files

- **`alternativeSQL.py`**: A script showcasing alternative process for creating tables in the database.
- **`assignment.txt`**: Contains the text from the original `README.md` file.
- **`optional.txt`**: Includes a discussion on topics mentioned in the optional assignment tasks.

## Note
The `requirements.txt` file was unchanged.