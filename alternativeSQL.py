import sqlite3

con = sqlite3.connect('alternative_data_storage.db')
control = con.cursor()

control.execute(
    '''
    CREATE TABLE IF NOT EXISTS housing (
        longitude REAL,
        latitude REAL,
        housing_median_age INTEGER,
        total_rooms INTEGER,
        total_bedrooms INTEGER,
        population INTEGER,
        households INTEGER,
        median_income REAL,
        ocean_proximity TEXT
    )
    '''    
)

data = []
control.executemany(
    '''
    INSERT INTO housing (longitude, latitude, housing_median_age, total_rooms, 
                         total_bedrooms, population, households, median_income, ocean_proximity)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
, data)

con.commit()
con.close()