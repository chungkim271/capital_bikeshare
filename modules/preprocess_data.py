import numpy as np
import pandas as pd
import os, requests
from pathlib import Path
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def trips_data(filepath):
    """
        Reads trips data and create: date, year, month, day, hour, week, holiday, weekend, duration_m 

    """

    trips = pd.read_csv(filepath, parse_dates=[1,2])

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
    trips['duration_m'] = trips['duration']/60
    trips.rename({
        "start_station_number":"start_station_id",
        "end_station_number":"end_station_id"
        }, axis=1, inplace=True)

    return(trips)
    

def master_data(datapath, save_to=None):

    """
    Creates "master" data by merging trips data in datapath and stations data.
    Reports trips that are not in (meta) stations data and drop them for now. 

    In addition to columns from trips and stations, the output data inlcudes:
        distance = euclidean distance between start and end stations

    """
    if isinstance(datapath, str):
        trips_filepaths = [Path(datapath, filepath) for filepath in sorted(os.listdir(datapath)) if filepath.endswith("tripdata.csv")]
    elif isinstance(datapath, list):
        trips_filepaths = datapath
    else:
        raise TypeError("Provide a str for data directory or a list of filepaths")

    trips  = pd.concat([trips_data(filepath) for filepath in trips_filepaths])
    stations = stations_data()

    df = pd.merge(trips, stations, how='left', left_on='start_station_id', right_on='station_id')

    not_in_stations = df.start_station_id[df.station_id.isna()].unique()
    num_missing = sum(df.station_id.isna())
    print("In trips(start) but not in stations: {}, {:,} trips total".format(not_in_stations, num_missing))
    print("Dropping trips that are not in stations data for now.")
    df = df.loc[~(df.station_id.isna()|df.start_station_id.isna()), :]

    df.drop('station_id', axis=1, inplace=True)
    df = pd.merge(df, stations, how='left', left_on='end_station_id', right_on='station_id',
             suffixes=['_start', '_end'])

    not_in_stations = df.end_station_id[df.station_id.isna()].unique()
    num_missing = sum(df.station_id.isna())
    print("In trips(end) but not in stations: {}, {:,} trips total".format(not_in_stations, num_missing))
    print("Dropping trips that not in stations data for now.")

    df = df.loc[~(df.station_id.isna()|df.end_station_id.isna()), :]

    # distance
    # euclidean distance between start and end stations. 
    # simple calculations using the parameters for DC Ben gave. 
    df['distance'] = np.sqrt(np.square((df.latitude_end - df.latitude_start)*111) + 
                             np.square((df.longitude_end - df.longitude_start)*85))

    if save_to:
        try:
            df.to_csv(save_to, sep=",")
        except:
            #### WRITE ERROR HANDLING
            pass

    return(df, stations)


def stations_data(api=True, filepath='data/raw/Capital_Bike_Share_Locations.csv'):
    """
    Cleans Capital Bikeshare stations data. The output data includes:
        
        station_id : TERMINAL_NUMBER in original data
        latitude
        longitude 
        capacity : the sum of "NUMBER OF EMPTY DOCKS" and "NUMBER OF BIKES" in original data
    
    The data can be downloaded manually from: https://opendata.dc.gov/datasets/capital-bike-share-locations
    
    """

    if api:
        stations = get_stations_data_from_opendc()
    else:
        stations = pd.read_csv(filepath)
    
    stations.columns = [col.replace(' ', '_', -1).lower() for col in stations.columns.values]
    
    stations.rename(columns = {"terminal_number":"station_id"}, inplace=True)
    stations['station_id'] = stations.station_id.astype(int)
    stations['capacity'] = stations['number_of_empty_docks'] + stations['number_of_bikes']
    to_keep = ["station_id", "address", "latitude", "longitude", "capacity"]
    stations = stations[to_keep]
    
    return(stations)


def get_stations_data_from_opendc():
    url = 'https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_WebMercator/MapServer/5/query?' \
        'where=1%3D1&outFields=ADDRESS,TERMINAL_NUMBER,LATITUDE,LONGITUDE,NUMBER_OF_BIKES,NUMBER_OF_EMPTY_DOCKS&' \
        'outSR=4326&f=json'

    r = requests.get(url=url)
    data = r.json()

    columns = data['fieldAliases'].keys()
    rows = [list(row['attributes'].values()) for row in data['features']]
    df = pd.DataFrame(rows, columns=columns)
    return(df)


def print_unique_values(df, show_limit=20):
    for col in df.columns:
        if df[col].dtypes == 'O':
            unique_values = df[col].unique()
            if len(unique_values) > show_limit:
                print("%s : %s unique values" % (col, len(unique_values)))
            else:
                print("%s : \n%s" % (col, df[col].value_counts()))
            print('---------------------')