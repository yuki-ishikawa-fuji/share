#!/usr/bin/env python
# coding: utf-8

## Import required modules
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_binary
import numpy as np
import pandas as pd
import datetime
import calendar
from dateutil.relativedelta import relativedelta
from os.path import isdir, isfile
import os
import time


## Define funcions
def scrapeStreamFlow(station_id, start_time, end_time):
    """
    Scraping streamflow data at a specified station during an arbitary designated period from 水文水質データベース
    水文水質データベースURL: http://www1.river.go.jp/
    Args:
        sation_id  (int): Measuring station ID
        start_time　　　 :  Data acquision start time (YYYY/mm)
        end_time　　　   : Data acquision end time (YYYY/mm)
    Returns:
        df_sf (pandas.dataFrame): A DataFrame of the scraping data
    """
    t0 = time.time()
    
    # URL (Desiganate ID, Start date, and End date)
    url = 'http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=7&ID={0}&BGNDATE={1}0131&ENDDATE={1}1231&KAWABOU=NO'
    # Generate a year list of target period
    year_list = [str(y) for y in range(start_time, end_time + 1)]
    # Loop for the year list
    sf_data_list = []
    for year in year_list:
        url_tmp = url.format(station_id, year)
        sf_data = subtractSFData(url_tmp, year)
        # Skip to the next loop if data does not exist
        if sf_data is None:
            continue
        sf_data_list += sf_data
    # Transform the list into DataFrame
    df_sf = list2dataframe(sf_data_list)
    
    # Save the DataFrame as CSV file
    file_name = f'daily_{station_id}_{start_time}-{end_time}.csv'
    outdir = './SUISUI-StreamflowData'
    if not isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '/' + file_name
    if not isfile(outfile):
        df_sf.to_csv(outfile)
        print(f'{file_name} has been processed')
    
    dt = time.time()-t0
    print(f'elapsed time: {dt:.2f} sec')
    
    return


def subtractSFData(url, year):
    """
    Subtract streamflow data from the URL in argument and return as a list.
    Args:
        url (String): the URL from which data is acquired
    Returns:
        sf_data_list (list): a list of streamflow data
    """
    try: 
        # Create the driver and operate the disignated URL page
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        # Acquire all the 'tr' tags from the table data
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.findAll("table")[1]
        rows = table.findAll("tr")
        driver.close()
    except:
        return None
    
    # Generate a streamflow data list from the table
    sf_list = []
    for row in rows[2:]:
        list_tmp = []
        td_list = row.findAll(['td'])
        for td in td_list:
            list_tmp.append(td.get_text().replace('\u3000', ''))
        sf_list.append(list_tmp)
    
    sf_data = [v for v in np.array(sf_list).reshape(1,-1)[0].tolist() if v != '']
    sf_time = []
    dt = datetime.datetime.strptime(year+"/01/01", "%Y/%m/%d")   
    dt_end = datetime.datetime.strptime(year+"/12/31", "%Y/%m/%d")
    end_flag = False
    while(not end_flag):
        sf_time.append(dt.strftime("%Y/%m/%d"))
        if(dt == dt_end):
                # End loop
                end_flag = True
        else: 
            # Update the date
            dt = dt + relativedelta(days=1)

    sf_data_list = [[t, d] for t, d in zip(sf_time, sf_data)]

    return sf_data_list


def list2dataframe(sf_list):
    """
    Convert the data acquired by scraping to Pandas DataFrame
    Args:
        sf_list (list): a list of streamflow data  #[date, steramflow] is assumed
    Returns:
        df_sf (pandas.DataFrame): streamflow data with the DataFrame format
    """
    # Tranform the list into the Pandas DataFrame
    df_sf = pd.DataFrame(sf_list, columns=['date', 'streamflow'])
    
    # Convert the data format of 'time' into datetime and 'streamflow' into float
    df_sf['date'] = df_sf['date'].astype('str').apply(str2datetime)
    df_sf['streamflow'] = pd.to_numeric(df_sf['streamflow'], errors='coerce')
    df_sf.set_index('date', inplace=True)
    df_sf.sort_index(inplace=True)
    
    return df_sf


def str2datetime(string):
    """
    Convert String-type time data into datetime-type
    Args:
        string (String): a string of date (YYYY/MM/DD)
    Returns:
        date (datetime.datetime): datetime-type of the given date
    """
    date = datetime.datetime.strptime(string, "%Y/%m/%d")
                    
    return date



## Data creation
if __name__ == '__main__':
    # Designate the period of data acquision
    start_time = 2017 # enter the start year in integer
    end_time = 2018 # enter the end year in integer
    # Specify the measuring station ID
    station_id = 305061285512010 # The station ID of Furi (布里)

    # Acquire the data
    scrapeStreamFlow(station_id, start_time, end_time)