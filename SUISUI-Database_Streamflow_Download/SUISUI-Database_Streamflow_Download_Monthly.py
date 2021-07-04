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
    url = 'http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=6&ID={0}&BGNDATE={1}&ENDDATE={2}&KAWABOU=NO'
    # Generate a year list of target period
    date_list = fetchDateList(start_time, end_time)
    # Loop for the year list
    sf_data_list = []
    for date in date_list:
        begin, end = date[0], date[1]
        url_tmp = url.format(station_id, begin, end)
        sf_data = subtractSFData(url_tmp)
        # Skip to the next loop if data does not exist
        if sf_data is None:
            continue
        sf_data_list += sf_data
    # Transform the list into DataFrame
    df_sf = list2dataframe(sf_data_list)
    
    # Save the DataFrame as CSV file
    file_name = f'{station_id}_{start_time[0:4]}{start_time[5:]}-{end_time[0:4]}{end_time[5:]}.csv'
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


def fetchDateList(start_time, end_time):
    """
    Create a list of time which consists of a set of start and end date for each month
    Args:
        start_time　　　 :  Data acquision start time (YY/mm)
        end_time　　　   : Data acquision end time (YY/mm)
    Returns:
        date_list (list): A list of time at an arbitary specifed period
    """
    # Convert String-type time data into datetime-type, and calculate the number of loop (the gap of monthes between start and end)
    dt_bgn = datetime.datetime.strptime(start_time+"/01", "%Y/%m/%d")
    dt_end = datetime.datetime.strptime(end_time+"/01", "%Y/%m/%d") + relativedelta(months=1, days=-1)
    
    end_flag = False
    date_list = []
    
    # Designate the start and end time for the first month of the target period
    search_bgn = dt_bgn
    search_end = search_bgn +  relativedelta(months=1, days=-1)
    
    while(not end_flag):
        date_list.append([search_bgn.strftime("%Y%m%d"), search_end.strftime("%Y%m%d")])
        if(search_end == dt_end):
            # End loop
            end_flag = True
        else: 
            # Update month
            search_bgn = search_bgn + relativedelta(months=1)
            search_end = search_bgn + relativedelta(months=1, days=-1)
    
    return date_list


def subtractSFData(url):
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
        list_tmp = [row.findAll(['th'])[0].get_text()]
        td_list = row.findAll(['td'])
        for td in td_list:
            list_tmp.append(td.get_text().replace('\u3000', ''))
        sf_list.append(list_tmp)
    
    sf_data = np.array(sf_list)[:, 1:].reshape(1,-1)[0].tolist()
    sf_time = []
    for date in np.array(sf_list)[:, 0]:
        for h in range(1,25):
            sf_time.append(f'{date} {h:02d}:00')

    sf_data_list = [[t, d] for t, d in zip(sf_time, sf_data)]

    return sf_data_list


def list2dataframe(sf_list):
    """
    Convert the data acquired by scraping to Pandas DataFrame
    Args:
        sf_list (list): a list of streamflow data  #[date&time, steramflow] is assumed
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
    if string[-5:] == '24:00':
        string = string[:-5] + ' ' + '00:00'
        date = datetime.datetime.strptime(string, "%Y/%m/%d %H:%M")
        date += datetime.timedelta(days=1)
    else:
        date = datetime.datetime.strptime(string, "%Y/%m/%d %H:%M")
    return date



## Data creation
if __name__ == "__main__":
    # Designate the period of data acquision
    start_time = '2000/01' # the format has to be 'YYYY/mm'
    end_time = '2000/12' # the format has to be 'YYYY/mm'
    # Specify the measuring station ID
    station_id = 303061283310070 # The station ID of Ochiai-bashi (落合橋)

    # Acquire the data
    scrapeStreamFlow(station_id, start_time, end_time)