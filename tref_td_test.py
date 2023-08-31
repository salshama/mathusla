#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mp
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

# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# Plot TD coarse vs TRef and plot average ToA in each 2D bin
# 
# 2d hist
# x axis = td coarse
# y axis = tref
# height or weight = avg toa

# In[16]:


files_tref = glob.glob('/eos/user/s/salshama/MATHUSLA/Tref_TD_Test/all_tref/Run*_list.txt')


# In[48]:


ToA_4  = []
ToA_4_std = []
ToA_22 = []
f_names = []

for file in files_tref:
    if len(get_data(file,'ToA')) > 0:
        ToA_4.append(np.average(get_data(file,'ToA')[4]))
        ToA_4_std.append(np.std(get_data(file,'ToA')[4]))
        f_names.append(file)
        #ToA_22.append(np.average(get_data(file,'ToA')[22]))


# In[49]:


f_names = [sub.replace('list','Info') for sub in f_names]


# In[50]:


def lines_that_contain(string,fp):
    return [line for line in fp if string in line]


# In[51]:


td_values = []
tref_values = []

for file in f_names:
    fn = open(file,'r')
    lines = fn.readlines()
    for line in lines:
        if 'TD_Coarse' in line:
            for td in line.split():
                if td.isdigit():
                    td_values.append(td)
        if 'TrefWindow' in line:
            for tref in line.split():
                if tref.isdigit():
                    if tref == '2' or tref == '1':
                        tref = int(tref)*1000
                    tref_values.append(tref)


# In[52]:


# avg_toa4 = np.mean(ToA_4)
# avg_toa22 = np.mean(ToA_22)


# In[53]:


# tref_all = (2000,1000,1000,800,600,400,200,150,100,1000,600,400,200,150,100,80,60,1000,800,600,400,200,150,100,80,60,1000,800,600,200,1000,800,400,200,150,100,80,60,950,850,750,700,680,670,660,650,640,630,620,590,1000,800,600,400)
# td_all = (250,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,180,180,180,180,180,180,180,180,180,180,180,150,150,150,150,170,170,170,170,170,170,170,170,160,160,160,160,160,160,160,160,160,160,200,200,200,200)


# In[54]:


tref_values = np.array(tref_values).astype(int)
td_values = np.array(td_values).astype(int)
ToA_4 = np.array(ToA_4).astype(int)
ToA_4_std = np.array(ToA_4_std).astype(float)


# In[74]:


ToA_4


# In[68]:


tref_values


# In[73]:


plt.figure(figsize=(7,7))
plt.hist2d(td_values[np.where(tref_values!=2000)],tref_values[np.where(tref_values!=2000)],bins=30,weights=ToA_4[np.where(tref_values!=2000)],cmin=1,vmax=150)

# for i in range(len(histy[2])-1):
#     for j in range(len(histy[1])-1):
#         plt.text(histy[j],histy[i], histy[0][i,j], color="w", ha="center", va="center", fontweight="bold")
#plt.hist2d(td_all,tref_all,weights=ToA_22[:54])
plt.colorbar(orientation='vertical')
plt.title('TD Coarse Threshold vs Tref Window')
plt.xlabel('TD Coarse Threshold')
#plt.yticks(np.linspace(60,1000))
plt.ylabel('Tref Window (ns)')
plt.tight_layout()
plt.savefig('tref_td_test')
plt.show()


# In[63]:


plt.figure(figsize=(7,7))
plt.hist2d(td_values,tref_values,bins=20,weights=ToA_4,cmin=1,vmax=150)

# for i in range(len(histy[2])-1):
#     for j in range(len(histy[1])-1):
#         plt.text(histy[j],histy[i], histy[0][i,j], color="w", ha="center", va="center", fontweight="bold")
#plt.hist2d(td_all,tref_all,weights=ToA_22[:54])
plt.colorbar(orientation='vertical')
plt.title('TD Coarse Threshold vs Tref Window')
plt.xlabel('TD Coarse Threshold')
plt.ylabel('Tref Window (ns)')
plt.tight_layout()
plt.savefig('tref_td_test')
plt.show()


# In[58]:


plt.figure(figsize=(7,7))
plt.hist2d(td_values,tref_values,bins=15,weights=ToA_4_std,cmin=1,vmax=150)

# for i in range(len(histy[2])-1):
#     for j in range(len(histy[1])-1):
#         plt.text(histy[j],histy[i], histy[0][i,j], color="w", ha="center", va="center", fontweight="bold")
#plt.hist2d(td_all,tref_all,weights=ToA_22[:54])
plt.colorbar(orientation='vertical')
plt.title('TD Coarse Threshold vs Tref Window')
plt.xlabel('TD Coarse Threshold')
plt.ylabel('Tref Window (ns)')
plt.tight_layout()
plt.savefig('tref_td_test_std')
plt.show()


# In[25]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_4[:11],td_500)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 500, Ch4')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref500_4')
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_4[:11],td_500)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 500, Ch4')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref500_4')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_22[:11],td_500)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 500, Ch22')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref500_22')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_4[:11],td_600)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 600, Ch4')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref600_4')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_22[:11],td_600)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 600, Ch22')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref600_22')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_4[:11],td_800)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 800, Ch4')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref800_4')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.hist2d(ToA_22[:11],td_800)
plt.colorbar(orientation='vertical')
plt.title('Tref Window = 800, Ch22')
plt.xlabel('ToA (ns)')
plt.ylabel('TD Coarse')
plt.tight_layout()
plt.savefig('tref800_22')
plt.show()


# In[ ]:


# Creating a function that plots the category type given a set of files

def plot_category(files_tref,tref_value,channel,category):
    
    plt.figure(figsize=(6,6))
    
    for run_num,file in enumerate(files_tref):
        
        data = get_data(file,category)[channel]
        
        if tref_value == 500:
            run_num += 2
        elif tref_value == 600:
            run_num += 13
        elif tref_value == 800:
            run_num += 24
        
        #entries,edges, _ = plt.hist(data,histtype='step',bins=np.linspace(min(data),max(data),20),label=run_num)
        entries,edges, _ = plt.hist(data,histtype='step',bins=np.linspace(50,70,100),label=run_num)

        bin_centers = 0.5*(edges[:-1]+edges[1:])
        plt.errorbar(bin_centers,entries,yerr=np.sqrt(entries)*5,capsize=1,fmt='r.')
        #plt.errorbar(bin_centers,entries,yerr=np.sqrt(entries)*10,capsize=1,fmt='r.')
        
    plt.title("Tref Window = {}, Channel = {}".format(tref_value,channel))
    plt.xlabel("ToA (ns)")
    plt.ylabel("Counts")
    plt.legend(loc=1)
    #plt.savefig("tref{}_ch{}.jpeg".format(tref_value,channel))
    plt.show()

