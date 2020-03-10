import numpy as np
import pandas as pd
import os
from pathlib import Path
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def trips_data(filepath):
    """
    """

    trips = pd.read_csv(filepath, parse_dates = [1,2])

    # standardize column names
    trips.columns = [col.replace(' ', '_', -1).lower() for col in trips.columns.values]

    # feature engineer date variables
    cal = calendar()
    holidays = cal.holidays(min(trips['start_date']), max(trips['start_date']))
    trips['date'] = trips['start_date'].dt.date
    trips['year'] = trips['start_date'].dt.year.astype(int)
    trips['month'] = trips['start_date'].dt.month.astype(int)
    trips['day'] = trips['start_date'].dt.dayofweek.astype(int)
    trips['hour'] = trips['start_date'].dt.hour.astype(int)
    trips['week'] = trips['start_date'].dt.week.astype(int)
    trips['holiday'] = trips['date'].astype('datetime64').isin(holidays)
    trips['weekend'] = trips['day'].isin([6, 7])
    trips.rename({
        "start_station_number":"start_station_id",
        "end_station_number":"end_station_id"
        }, axis=1, inplace=True)

    return(trips)
    

def master_data(datapath, save_to=None):

    """
    Creates "master" data by merging trips data in datapath and stations data.
    Reports stations that exist in only left or only right dataframe and drops those trips. 


        

    """
    if isinstance(datapath, str):
        trips_filepaths = [Path(datapath, filepath) for filepath in sorted(os.listdir(datapath)) if filepath.endswith("tripdata.csv")]
    elif isinstance(datapath, list):
        trips_filepaths = datapath
    else:
        raise TypeError("Provide a str for data directory or a list of filepaths")

    trips  = pd.concat([trips_data(filepath) for filepath in trips_filepaths])
    stations = stations_data()

    df = pd.merge(trips, stations, how='outer', left_on='start_station_id', right_on='station_id')

    not_in_start_trips = df.station_id[df.start_station_id.isna()].unique()
    not_in_stations = df.start_station_id[df.station_id.isna()].unique()
    print("In trips(start) but not in stations: %s" % list(not_in_stations))
    print("In stations but not in trips(start) %s" % list(not_in_start_trips))

    df = df.loc[~(df.station_id.isna()|df.start_station_id.isna()), :]

    df.drop('station_id', axis=1, inplace=True)
    df = pd.merge(df, stations, how='outer', left_on='end_station_id', right_on='station_id',
             suffixes=['_start', '_end'])

    not_in_end_trips = df.station_id[df.end_station_id.isna()].unique()
    not_in_stations = df.end_station_id[df.station_id.isna()].unique()
    print("In trips(end) but not in stations: %s" % list(not_in_stations))
    print("In stations but not in start %s" % list(not_in_end_trips))

    df = df.loc[~(df.station_id.isna()|df.end_station_id.isna()), :]

    # distance
    # euclidean distance between start and end stations 
    df['distance'] = np.sqrt(np.square((df.latitude_end - df.latitude_start)*111) + 
                             np.square((df.longitude_end - df.longitude_start)*85))

    if save_to != None:
        try:
            df.to_csv(save_to, sep=",")
        except:
            pass

    return(df)




def stations_data(filepath='data/Capital_Bike_Share_Locations.csv'):
    """
    Cleans Capital Bikeshare locaitons data. This output meta data includes:
        
        station_id : TERMINAL_NUMBER in original data
        latitude
        longitude 
        capacity : the sum of "NUMBER OF EMPTY DOCKS" and "NUMBER OF BIKES" in original data
    
    The data is downloaded from: https://opendata.dc.gov/datasets/capital-bike-share-locations
    
    """
    stations = pd.read_csv(filepath)
    stations.columns = [col.replace(' ', '_', -1).lower() for col in stations.columns.values]
    
    stations['capacity'] = stations['number_of_empty_docks'] + stations['number_of_bikes']
    stations.rename(columns = {"terminal_number":"station_id"}, inplace=True)
    to_keep = ["station_id", "latitude", "longitude", "capacity"]
    stations = stations[to_keep]
    
    # early analysis showed that there's a station number 31202 in trips data 
    stations.loc[stations.shape[0], :] = [31202 , 38.912638, -77.032008, np.NaN]
    return(stations)