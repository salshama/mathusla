#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import statistics
#plt.style.use('science')

"""
This module is designed to deal with the Janus output datafiles

get_data() can be used to get a single data column
All_info() gets all data that is given in a file
data_dict_maker() puts the output of All_info() into a more usable form

-- As of June 26, 2022 --
These are the most up-to-date versions of these functions
Any code which uses these functions (or variations) from another module should 
be updated when possible.
All new code should import this module for reading Janus files.  
"""


# <h3>Janus Data Extractor</h3>

# In[2]:


def order(file):
    '''
    Order is the default way that files should be sorted in a list (by run #)
    An alternative use is getting the run number from a file

    Parameters
    ----------
    a : str
        The file name

    Returns
    -------
    run_num : int
        The run number of the file

    '''
    return int(file.split('Run')[1].split('_')[0])


# In[3]:


def index_manager(line, column):
    '''
    Gives the index that the get_data() function will use
    This function should not be called by itself
    
    Parameters
    ----------
    line : list[str]
        The entries of the file line
    column : str
        The desired column.  Options: LG, HG, ToA, ToT

    Returns
    -------
    ind : int
        The index corresponding to the desired column
    chan_col : int
        The column index for the given file type

    '''
    if 'ToA_LSB' in line:
        ind = {'ToA':-2, 'LG':-4,'HG':-3, 'ToT':-1}[column]
        chan_col = -5
    else:
        ind = {'HG':-1, 'LG':-2}[column]
        chan_col = -3
    
    return ind, chan_col


# In[4]:


def get_data(file:str,column:str, LSB = True):
    '''
    Returns a dictionary of the wanted data sorted by channel number
    Call this function if you need one column of data from a run

    Parameters
    ----------
    file : str
        The file containing the data
    column : str
        Which column you want.  Options: LG, HG, ToA, ToT 
    LSB : TYPE, optional
        For timing file, is the data GIVEN in LSB or is it in ns. This
        parameter does not affect the output, which is always in ns.
        The default is True.

    Returns
    -------
    data : dict
        A dictionary is returned where the keys are the channel numbers.
        Each dictionary entry is a numpy array 
    '''
    
    data = {}
    
    with open(file) as f:
        
        for i,line in enumerate(f):
            line = line.strip()
            
            #This catches the header line
            #Identifying the number of columns tells if file includes timing
            if line[0] == 'T':
                
                ind, chan_col = index_manager(line.split(), column)
            
            #This eliminates the preamble lines
            elif not(line[0] == '/'):
                
                #Splits the data into a usable form
                info = line.split()
                chan = int(info[chan_col])
                
                #Sometimes ToA is not collected
                #This Catches Those
                if info[ind].isnumeric():
                    pt = float(info[ind])
                    
                    if (column == 'ToA' or column == 'ToT') and LSB: 
                        pt = pt / 2
                        
                    #Checks if that channel has already been recorded
                    #If so, it appends to the list
                    #Otherwise it makes a new one
                    if chan in data.keys():
                        data[chan] = np.append(data[chan],np.array([pt]))
                    else:
                        data[chan] = np.array([pt])
               
    return data


# In[5]:


def All_info(file, unpack = False, timing = True):
    '''
    This function extracts all of the events from the datafile.
    Returns array with a row as each event.
    Column order is Time | Channel | LG | HG | ToA (if possible) | ToT (if possible)
    It is recommended that the array returned by this function is passed to data_dict_maker()
    to clean it up.
    
    Parameters
    ----------
    file : str
        The file containing the data
    unpack: bool
        This transposes the data being returned in the numpy array, giving each data
        category in the rows, not each event.  It allows for the calling:
        >>> time, chan, LG, HG = All_info_no_timing(filename, True)
        Default is False.

    Returns
    -------
    data : numpy.ndarray
        A two dimensional array containing all data entries from the file.  By 
        default, each event is a row with 6 columns
    '''
    all_data = []
    data_prev = []
    
    with open(file) as f:
        for i,line in enumerate(f):
            
            line = line.strip()
            
            if not(line[0] == '/' or line[0] == 'T'):
                raw_data = line.split()
                
                data = data_form(raw_data,data_prev, timing)
                
                if data[-1] != '-' and timing:
                    data[-1], data[-2] = data[-1]/2,data[-2]/2 
                
                    all_data.append(data)
                    
                elif not(timing):
                    all_data.append(data)
                    
                else:
                    pass #print(f'No Timing on row {i}')
                    
                if raw_data[0] != '00':
                    data_prev = data
            
    if unpack:
        return np.array(all_data).T           
    else:
        return np.array(all_data)


# In[6]:


def data_form(data,prev_data, timing):
    '''Deals with gaps in data file for All_info()
    Manages the problem of not all rows having time stamps
    
    Parameters
    ----------
    data: list[str]
        The row of data you want to append
    
    data_prev: list[str]
        The last row of data that had a timestamp (if data has a timestamp, this is ignored)
    
    timing: bool
        If this is a spect_timing file.  Used for indexing the file
    
    Returns
    -------
    data: numpy.ndarray
        A 1 by n array containing the wanted data
    '''
    if not(timing):
        chan, LG, HG = data[-3:]
    else:
        chan, LG, HG, ToA, ToT = data[-5:]
        if ToA == '-':
            return '-'
    
    if data[0] != '00':
        T_stamp = data[0]
    else:
        T_stamp = prev_data[0]
        
    if not(timing):
        return np.array([T_stamp,chan, LG, HG]).astype('float')
    
    else:
        return np.array([T_stamp,chan, LG, HG, ToA, ToT]).astype('float')


# In[7]:


def data_dict_maker(time,channel,LG,HG, ToA = None, ToT = None):
    '''
    Formats data into a nice dictionary format for readablility
    
    Parameters
    ----------
    time : numpy.ndarray
        Contains Times
    channel : numpy.ndarray
        Contains channel
    LG : numpy.ndarray
        Low Gain ADC Counts
    HG : numpy.ndarray
        High Gain ADC Counts
    ToA : numpy.ndarray, optional
        Time of Arrival.  Default is none.
    ToT : numpy.ndarray, optional
        Time over Threshold.  Default is none.

    Returns
    -------
    data : dict
        Contains keys for each column input 
    '''
    data = {}
    #np.unique eliminates doubles, taking the length gives the number of channels
    num_channels = len(np.unique(channel))
    
    #Only want the time for each event, not every channel entry
    data['t'] = time[::num_channels]
    #exp_t is the more useful time information
    #It sets the experiment start time as 0
    data['exp_t'] = data['t'] - data['t'][0]
    
    for chan in np.unique(channel):
        data[f'LG{int(chan)}'] = LG[channel == chan]
        data[f'HG{int(chan)}'] = HG[channel == chan]
        print(ToA)
        if type(ToA) != type(None):
            data[f'ToA{int(chan)}'] = ToA[channel == chan]
        if type(ToT) != type(None):
            data[f'ToT{int(chan)}'] = ToT[channel == chan]

    return data


# In[8]:


"""
Functions that might be useful for Sarah
"""
def All_info(file, unpack = False):
    '''
    This function extracts all of the events from the datafile
    Returns array with a row as each event 
    Column order is Time | Channel | LG | HG | ToA | ToT
    Unpack switches to each row is a different quantity
    '''
    all_data = []
    data_prev = []
    with open(file) as f:
        for i,line in enumerate(f):
            
            line = line.strip()
            
            if not(line[0] == '/' or line[0] == 'T'):
                data = line.split()
                
                data = data_form(data,data_prev)
                
                if data[0] != '-':
                    data[-1], data[-2] = data[-1]/2,data[-2]/2 
                    
                    all_data.append(data)
                    data_prev = data
                    
                else:
                    print(f'No timing on row {i}')

    if unpack:
        return np.array(all_data).T
    
    else:
        return np.array(all_data)


# <h3>Janus Data Plotter</h3>

# In[9]:


# """
# Plot Creator For A Data File\
    
# This File follows the general procedure:
# 1. You call the function to make your plot
# 2. That function calls it's corresponding data function
# 3. The data function calls get_data() to extract the right columns

# get_data() can be used to get a specific column from an list file

# "category"_data() uses get_data() for multiple files, then returns the mean and 
# standard deviation of each file

# Created on Wed May 10 20:10:03 2023

# @author: Branden Aitken
# """

# def order(a):
#     return int(a.split('Run')[1].split('_')[0])

# def get_data(file,category):
    
#     ind = {'ToA':-2, 'LG':-4,'HG':-3, 'ToT':-1}[category]
    
#     data = {4:[],22:[]}
    
#     with open(file) as f:
        
#         for i,line in enumerate(f):
            
#             #Takes out header
#             if i >= 9:
                
#                 info = line.split()
#                 chan = int(info[-5])
                
#                 #Sometimes ToA is not collected
#                 #This Catches Those
#                 try:
#                     pt = int(info[ind])
                    
#                     if ind > -3: #Because of structure
#                         pt = pt / 2
                        
#                     data[chan].append(pt)
                    
#                 except ValueError:
#                     pass #print(f'Line {i} Excluded From Sample')
               
#     return np.array(data[4]),np.array(data[22])


# In[10]:


def ToA_data(files):
    
    #Sets up storage system
    means = {4:[], 22:[]}
    rmss = {4:[], 22:[]}
    
    for f in files:
        print(f)
        chan4, chan22 = get_data(f,'ToA')
        
        if list(chan4) != []: #This eliminates runs with no data
            means[4].append(np.mean(chan4))
            rmss[4].append(np.std(chan4))
            #rmss[4].append(np.sqrt(np.mean(chan4**2) / len(chan4)))
            
        else:
            print(f'No Channel 4 Data in {f}')
        
        if list(chan22) != []:
            means[22].append(np.mean(chan22))
            rmss[22].append(np.std(chan22))
            #rmss[22].append(np.sqrt(np.mean(chan22**2) / len(chan22)))
        
        else:
            print(f'No Channel 22 Data in {f}')
        
    return means,rmss


# In[11]:


def get_ToA_plot(wildcard, exclude, file_out, cable):
    
    #Sorts out filenames
    
    filenames = glob.glob(wildcard)
    
    for file in filenames:
        if order(file) in exclude:
            filenames.remove(file)
    
    filenames = sorted(filenames, key = order)
    
    avg, sd = ToA_data(filenames)
    
    #This makes the plots 
    
    x = 30 * 2/3 * np.array(avg[4])
    
    fig, [plot1,plot2] = plt.subplots(1,2, figsize = (12,5),constrained_layout = True)
    plt.suptitle(cable)
    
    plot1.scatter(x, np.array(avg[4]), label = 'Channel 4')
    plot1.scatter(x, np.array(avg[22]), label = 'Channel 22')
    
    plot1.legend(loc = 8, frameon = True)
    plot1.grid()
    
    plot1.set_xlabel('Distance (cm)')
    plot1.set_ylabel('ToA (ns)')
    plot1.set_title('Time of Arrival vs Distance')
    
    plot2.scatter(x,np.array(sd[4]), label = 'Channel 4')
    plot2.scatter(x,np.array(sd[22]), label = 'Channel 22')
    
    plot2.legend(frameon = True)
    plot2.grid()
    
    plot2.set_xlabel('Distance (cm)')
    plot2.set_ylabel('Standard Deviation (ns)')
    plot2.set_title('Standard Deviation vs Distance')
    
    plt.savefig(file_out)
    
    return file_out in glob.glob(file_out)


# In[12]:


def ToT_data(files):
    
    #Sets up storage system
    meanToT = {4:[], 22:[]}
    meanADC = {4:[], 22:[]}
    
    for f in files:
        print(f)
        ToT4, ToT22 = get_data(f,'ToT')
        ADC4, ADC22 = get_data(f,'LG')
        
        if list(ToT4) != []: #This eliminates runs with no data
            meanADC[4].append(np.mean(ADC4))
            meanToT[4].append(np.mean(ToT4))
            
        else:
            print(f'No Channel 4 Data in {f}')
        
        if list(ToT22) != []:
            meanADC[22].append(np.mean(ADC22))
            meanToT[22].append(np.mean(ToT22))
            
        else:
            print(f'No Channel 22 Data in {f}')
        
    return meanADC, meanToT


# In[13]:


def get_ToT_plot(wildcard, exclude, file_out, cable):
    filenames = glob.glob(wildcard)
    
    for file in exclude:
        f1,f2 = file.split('*')
        filenames.remove(f1 + str(file) + f2)
    
    filenames = sorted(filenames,key = order) #Sorts filenames
    
    ADC,ToT = ToT_data(filenames)
    
    plt.figure(figsize = (6,4))
    plt.scatter(ADC[4],ToT[4], label = 'Channel 4')
    plt.scatter(ADC[22],ToT[22], label = 'Channel 22')
    
    
    plt.legend(frameon = True)
    plt.grid()
    
    plt.xlabel('ADC Counts')
    plt.ylabel('ToT (ns)')
    plt.title(f'Time over Threshold vs ADC Counts For {cable}')
    
    plt.savefig(file_out)
    
    return file_out in glob.glob(file_out)


# In[14]:


def Counts_ToA_data(files, gain):
    
    #Sets up storage system
    meanToA = {4:[], 22:[]}
    meanG = {4:[], 22:[]}
    
    for f in files:
        print(f)
        ToA4,ToA22 = get_data(f,'ToA')
        G4, G22 = get_data(f,gain)
        
        if list(ToA4) != [] and list(ToA22) != []:#This eliminates runs with no data
            meanG[4].append(np.mean(G4))
            meanToA[4].append(np.mean(ToA4))
            meanG[22].append(np.mean(G22))
            meanToA[22].append(np.mean(ToA22))
            
        else:
            print(f'Omit This file: {f}')
        
    return meanG, meanToA


# In[15]:


def get_Counts_ToA_plot(wildcard, exclude, file_out, cable,gain = 'LG'):
    '''
    Plots the average ToT for each file found in the wildcard against
    the ADC counts.
    Excluded files are files that appear in the wildcard, but you
    don't want to plot.  Give a list of run Numbers.
    file_out allows a plot to be saved
    fibre is the spot to put your fibre title
    >>> get_ToT_plot('May11th/Fibre 24/*_list.txt',[2,3,4], 'example.png', 'Fibre 24')
    '''
    
    filenames = glob.glob(wildcard)
    
    for file in filenames:
        if order(file) in exclude:
            filenames.remove(file)
    
    filenames = sorted(filenames,key = order) #Sorts filenames
    
    LG, ToA = Counts_ToA_data(filenames,gain)
    
    plt.figure(figsize = (6,4))
    plt.scatter(LG[4],ToA[4], label = 'Channel 4')
    plt.scatter(LG[22],ToA[22], label = 'Channel 22')
    
    
    plt.legend(frameon = True)
    plt.grid()
    
    if gain == 'LG':
        expand = 'Low Gain'
    else:
        expand = 'High Gain'
    
    plt.xlabel(f'{expand} Spect Mean')
    plt.ylabel('ToA (ns)')
    plt.title(f'ToA vs {expand} Spect Mean For {cable}')
    
    plt.savefig(file_out)
    
    return file_out in glob.glob(file_out)


# <h3>Analyzing Data</h3>

# In[16]:


no_crossing_lg    = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/0_crossing/Run10_list.txt','LG')
no_crossing_hg    = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/0_crossing/Run10_list.txt','HG')

one_crossing_lg   = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/1_crossing/Run11_list.txt','LG')
one_crossing_hg   = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/1_crossing/Run11_list.txt','HG')

two_crossing_lg   = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/2_crossing/Run12_list.txt','LG')
two_crossing_hg   = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/2_crossing/Run12_list.txt','HG')

three_crossing_lg = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/3_crossing/Run13_list.txt','LG')
three_crossing_hg = get_data('/eos/user/s/salshama/MATHUSLA/Fibre_Leakage_Test/3_crossing/Run13_list.txt','HG')


# In[17]:


ch4_0c_lg  = no_crossing_lg[4]
ch4_0c_hg  = no_crossing_hg[4]
ch22_0c_lg = no_crossing_lg[22]
ch22_0c_hg = no_crossing_hg[22]

ch4_1c_lg  = one_crossing_lg[4]
ch4_1c_hg  = one_crossing_hg[4]
ch22_1c_lg = one_crossing_lg[22]
ch22_1c_hg = one_crossing_hg[22]

ch4_2c_lg  = two_crossing_lg[4]
ch4_2c_hg  = two_crossing_hg[4]
ch22_2c_lg = two_crossing_lg[22]
ch22_2c_hg = two_crossing_hg[22]

ch4_3c_lg  = one_crossing_lg[4]
ch4_3c_hg  = one_crossing_hg[4]
ch22_3c_lg = one_crossing_lg[22]
ch22_3c_hg = one_crossing_hg[22]


# In[18]:


# Plots, error bars and standard deviation - by configuration

entries_40clg, edges_40clg, _ = plt.hist(ch4_0c_lg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
#entries_40chg, edges_40chg, _ = plt.hist(ch4_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4 HG')
bin_centers_40clg = 0.5*(edges_40clg[:-1]+edges_40clg[1:])
#bin_centers_40chg = 0.5*(edges_40chg[:-1]+edges_40chg[1:])

entries_220clg, edges_220clg, _ = plt.hist(ch22_0c_lg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
#entries_220chg, edges_220chg, _ = plt.hist(ch22_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22 HG')
bin_centers_220clg = 0.5*(edges_220clg[:-1]+edges_220clg[1:])
#bin_centers_220chg = 0.5*(edges_220chg[:-1]+edges_220chg[1:])

plt.errorbar(bin_centers_40clg,entries_40clg,yerr=statistics.stdev(entries_40clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_40chg,entries_40chg,yerr=statistics.stdev(entries_40chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_220clg,entries_220clg,yerr=statistics.stdev(entries_220clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_220chg,entries_220chg,yerr=statistics.stdev(entries_220chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("No crossing (LG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('no_cross_lg')
plt.show()

entries_40chg, edges_40chg, _ = plt.hist(ch4_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
bin_centers_40chg = 0.5*(edges_40chg[:-1]+edges_40chg[1:])

entries_220chg, edges_220chg, _ = plt.hist(ch22_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
bin_centers_220chg = 0.5*(edges_220chg[:-1]+edges_220chg[1:])

plt.errorbar(bin_centers_40chg,entries_40chg,yerr=statistics.stdev(entries_40chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_220chg,entries_220chg,yerr=statistics.stdev(entries_220chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("No crossing (HG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('no_cross_hg')
plt.show()


# In[19]:


entries_41clg, edges_41clg, _ = plt.hist(ch4_1c_lg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
#entries_41chg, edges_41chg, _ = plt.hist(ch4_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4 HG)')
bin_centers_41clg = 0.5*(edges_41clg[:-1]+edges_41clg[1:])
#bin_centers_41chg = 0.5*(edges_41chg[:-1]+edges_41chg[1:])

entries_221clg, edges_221clg, _ = plt.hist(ch22_1c_lg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
#entries_221chg, edges_221chg, _ = plt.hist(ch22_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22 HG')
bin_centers_221clg = 0.5*(edges_221clg[:-1]+edges_221clg[1:])
#bin_centers_221chg = 0.5*(edges_221chg[:-1]+edges_221chg[1:])

plt.errorbar(bin_centers_41clg,entries_41clg,yerr=statistics.stdev(entries_41clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_41chg,entries_41chg,yerr=statistics.stdev(entries_41chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_221clg,entries_221clg,yerr=statistics.stdev(entries_221clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_221chg,entries_221chg,yerr=statistics.stdev(entries_221chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("One crossing (LG, 21mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('one_cross_lg')
plt.show()

entries_41chg, edges_41chg, _ = plt.hist(ch4_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
bin_centers_41chg = 0.5*(edges_41chg[:-1]+edges_41chg[1:])

entries_221chg, edges_221chg, _ = plt.hist(ch22_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
bin_centers_221chg = 0.5*(edges_221chg[:-1]+edges_221chg[1:])

plt.errorbar(bin_centers_41chg,entries_41chg,yerr=statistics.stdev(entries_41chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_221chg,entries_221chg,yerr=statistics.stdev(entries_221chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("One crossing (HG, 21mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('one_cross_hg')
plt.legend()
plt.show()


# In[20]:


entries_42clg, edges_42clg, _ = plt.hist(ch4_2c_lg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
#entries_42chg, edges_42chg, _ = plt.hist(ch4_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4 HG')
bin_centers_42clg = 0.5*(edges_42clg[:-1]+edges_42clg[1:])
#bin_centers_42chg = 0.5*(edges_42chg[:-1]+edges_42chg[1:])

entries_222clg, edges_222clg, _ = plt.hist(ch22_2c_lg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
#entries_222chg, edges_222chg, _ = plt.hist(ch22_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22 HG')
bin_centers_222clg = 0.5*(edges_222clg[:-1]+edges_222clg[1:])
#bin_centers_222chg = 0.5*(edges_222chg[:-1]+edges_222chg[1:])

plt.errorbar(bin_centers_42clg,entries_42clg,yerr=statistics.stdev(entries_42clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_42chg,entries_42chg,yerr=statistics.stdev(entries_42chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_222clg,entries_222clg,yerr=statistics.stdev(entries_222clg),uplims=True,lolims=True,fmt='r.',capsize=2)
#plt.errorbar(bin_centers_222chg,entries_222chg,yerr=statistics.stdev(entries_222chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Two crossings (LG, 33mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('two_cross_lg')
plt.show()

entries_42chg, edges_42chg, _ = plt.hist(ch4_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
bin_centers_42chg = 0.5*(edges_42chg[:-1]+edges_42chg[1:])

entries_222chg, edges_222chg, _ = plt.hist(ch22_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
bin_centers_222chg = 0.5*(edges_222chg[:-1]+edges_222chg[1:])

plt.errorbar(bin_centers_42chg,entries_42chg,yerr=statistics.stdev(entries_42chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_222chg,entries_222chg,yerr=statistics.stdev(entries_222chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Two crossings (HG, 33mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('two_cross_hg')
plt.show()


# In[21]:


entries_43clg, edges_43clg, _ = plt.hist(ch4_3c_lg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
# entries_43chg, edges_43chg, _ = plt.hist(ch4_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
bin_centers_43clg = 0.5*(edges_43clg[:-1]+edges_43clg[1:])
# bin_centers_43chg = 0.5*(edges_43chg[:-1]+edges_43chg[1:])

entries_223clg, edges_223clg, _ = plt.hist(ch22_3c_lg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
# entries_223chg, edges_223chg, _ = plt.hist(ch22_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
bin_centers_223clg = 0.5*(edges_223clg[:-1]+edges_223clg[1:])
# bin_centers_223chg = 0.5*(edges_223chg[:-1]+edges_223chg[1:])

plt.errorbar(bin_centers_43clg,entries_43clg,yerr=statistics.stdev(entries_43clg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_223clg,entries_223clg,yerr=statistics.stdev(entries_223clg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Three crossings (LG, 100mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('three_cross_lg')
plt.show()

entries_43chg, edges_43chg, _ = plt.hist(ch4_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='Ch4')
bin_centers_43chg = 0.5*(edges_43chg[:-1]+edges_43chg[1:])

entries_223chg, edges_223chg, _ = plt.hist(ch22_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='CH22')
bin_centers_223chg = 0.5*(edges_223chg[:-1]+edges_223chg[1:])

plt.errorbar(bin_centers_43chg,entries_43chg,yerr=statistics.stdev(entries_43chg),uplims=True,lolims=True,fmt='r.',capsize=2)
plt.errorbar(bin_centers_223chg,entries_223chg,yerr=statistics.stdev(entries_223chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Three crossings (HG, 100mm of tape)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('three_cross_hg')
plt.show()


# In[22]:


# Plots, error bars and standard deviation - by channel 4 (LG)

entries_40clg, edges_40clg, _ = plt.hist(ch4_0c_lg,bins=np.linspace(70,120,25),histtype='step',label='0x (LG)')
bin_centers_40clg = 0.5*(edges_40clg[:-1]+edges_40clg[1:])


plt.errorbar(bin_centers_40clg,entries_40clg,yerr=statistics.stdev(entries_40clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_41clg, edges_41clg, _ = plt.hist(ch4_1c_lg,bins=np.linspace(70,120,25),histtype='step',label='1x (LG)')

bin_centers_41clg = 0.5*(edges_41clg[:-1]+edges_41clg[1:])

plt.errorbar(bin_centers_41clg,entries_41clg,yerr=statistics.stdev(entries_41clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_42clg, edges_42clg, _ = plt.hist(ch4_2c_lg,bins=np.linspace(70,120,25),histtype='step',label='2x (LG)')
bin_centers_42clg = 0.5*(edges_42clg[:-1]+edges_42clg[1:])

plt.errorbar(bin_centers_42clg,entries_42clg,yerr=statistics.stdev(entries_42clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_43clg, edges_43clg, _ = plt.hist(ch4_3c_lg,bins=np.linspace(70,120,25),histtype='step',label='3x (LG)')
bin_centers_43clg = 0.5*(edges_43clg[:-1]+edges_43clg[1:])

plt.errorbar(bin_centers_43clg,entries_43clg,yerr=statistics.stdev(entries_43clg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Channel 4 (All Configurations - LG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('ch4_all_config_lg')
plt.show()

# # Plots, error bars and standard deviation - by channel 4 (HG)

entries_40chg, edges_40chg, _ = plt.hist(ch4_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='0x (HG)')
bin_centers_40chg = 0.5*(edges_40chg[:-1]+edges_40chg[1:])

plt.errorbar(bin_centers_40chg,entries_40chg,yerr=statistics.stdev(entries_40chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_41chg, edges_41chg, _ = plt.hist(ch4_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='1x (HG)')
bin_centers_41chg = 0.5*(edges_41chg[:-1]+edges_41chg[1:])

plt.errorbar(bin_centers_41chg,entries_41chg,yerr=statistics.stdev(entries_41chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_42chg, edges_42chg, _ = plt.hist(ch4_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='2x (HG)')
bin_centers_42chg = 0.5*(edges_42chg[:-1]+edges_42chg[1:])

plt.errorbar(bin_centers_42chg,entries_42chg,yerr=statistics.stdev(entries_42chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_43chg, edges_43chg, _ = plt.hist(ch4_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='3x (HG)')
bin_centers_43chg = 0.5*(edges_43chg[:-1]+edges_43chg[1:])

plt.errorbar(bin_centers_43chg,entries_40chg,yerr=statistics.stdev(entries_43chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Channel 4 (All Configurations - HG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('ch4_all_config_hg')
plt.show()

# Plots, error bars and standard deviation - by channel 22 (LG)

entries_220clg, edges_220clg, _ = plt.hist(ch22_0c_lg,bins=np.linspace(70,120,25),histtype='step',label='0x (LG)')
bin_centers_220clg = 0.5*(edges_220clg[:-1]+edges_220clg[1:])

plt.errorbar(bin_centers_220clg,entries_220clg,yerr=statistics.stdev(entries_220clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_221clg, edges_221clg, _ = plt.hist(ch22_1c_lg,bins=np.linspace(70,120,25),histtype='step',label='1x (LG)')
bin_centers_221clg = 0.5*(edges_221clg[:-1]+edges_221clg[1:])

plt.errorbar(bin_centers_221clg,entries_221clg,yerr=statistics.stdev(entries_221clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_222clg, edges_222clg, _ = plt.hist(ch22_2c_lg,bins=np.linspace(70,120,25),histtype='step',label='2x (LG)')
bin_centers_222clg = 0.5*(edges_222clg[:-1]+edges_222clg[1:])

plt.errorbar(bin_centers_222clg,entries_222clg,yerr=statistics.stdev(entries_222clg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_223clg, edges_223clg, _ = plt.hist(ch22_3c_lg,bins=np.linspace(70,120,25),histtype='step',label='3x (LG)')
bin_centers_223clg = 0.5*(edges_223clg[:-1]+edges_223clg[1:])

plt.errorbar(bin_centers_223clg,entries_223clg,yerr=statistics.stdev(entries_223clg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Channel 22 (All Configurations - LG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('ch22_all_config_lg')
plt.show()

# Plots, error bars and standard deviation - by channel 22 (HG)

entries_220chg, edges_220chg, _ = plt.hist(ch22_0c_hg,bins=np.linspace(70,120,25),histtype='step',label='0x (HG)')
bin_centers_220chg = 0.5*(edges_220chg[:-1]+edges_220chg[1:])

plt.errorbar(bin_centers_220chg,entries_220chg,yerr=statistics.stdev(entries_220chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_221chg, edges_221chg, _ = plt.hist(ch22_1c_hg,bins=np.linspace(70,120,25),histtype='step',label='1x (HG)')
bin_centers_221chg = 0.5*(edges_221chg[:-1]+edges_221chg[1:])

plt.errorbar(bin_centers_221chg,entries_221chg,yerr=statistics.stdev(entries_221chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_222chg, edges_222chg, _ = plt.hist(ch22_2c_hg,bins=np.linspace(70,120,25),histtype='step',label='2x (HG)')
bin_centers_222chg = 0.5*(edges_222chg[:-1]+edges_222chg[1:])

plt.errorbar(bin_centers_220chg,entries_220chg,yerr=statistics.stdev(entries_220chg),uplims=True,lolims=True,fmt='r.',capsize=2)

entries_223chg, edges_223chg, _ = plt.hist(ch22_3c_hg,bins=np.linspace(70,120,25),histtype='step',label='3x (HG)')
bin_centers_223chg = 0.5*(edges_223chg[:-1]+edges_223chg[1:])

plt.errorbar(bin_centers_223chg,entries_223chg,yerr=statistics.stdev(entries_223chg),uplims=True,lolims=True,fmt='r.',capsize=2)

plt.title("Channel 22 (All Configurations - HG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('ch22_all_config_hg')
plt.show()

