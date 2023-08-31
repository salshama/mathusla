#!/usr/bin/env python
# coding: utf-8

# In[29]:


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import statistics
#plt.style.use('science')


# <h3>Janus Data Extractor</h3>

# In[30]:


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


# In[31]:


def All_info_no_timing(file, unpack = False):
    
    '''
    This function extracts all of the events from the datafile
    Returns array with a row as each event 
    Column order is Time | Channel | LG | HG |
    Unpack switches to each row is a different quantity
    '''
    
    all_data = []
    data_prev = []
    with open(file) as f:
        for i,line in enumerate(f):
            
            line = line.strip()
            
            if not(line[0] == '/' or line[0] == 'T'):
                data = line.split()
                
                data = data_form(data,data_prev, no_timing = True)
                
                data_prev = data 
                
                all_data.append(data)

    if unpack:
        return np.array(all_data).T           
    
    else:
        return np.array(all_data)


# In[32]:


def data_form(data,prev_data, no_timing):
    '''Deals with gaps in data file for All_info()'''
    
    if no_timing:
        chan, LG, HG = data[-3:]
    else:
        chan, LG, HG, ToA, ToT = data[-5:]
    
    if chan == '04':
        T_stamp = data[0]
    else:
        T_stamp = prev_data[0]
     
    if ToA == '-':
        return '-'
    
    return np.array([T_stamp,chan, LG, HG, ToA, ToT]).astype('float')


# <h3>Janus Data Plotter</h3>

# In[33]:


"""
Plot Creator For A Data File
    
This File follows the general procedure:
1. You call the function to make your plot
2. That function calls it's corresponding data function
3. The data function calls get_data() to extract the right columns

get_data() can be used to get a specific column from an list file

"category"_data() uses get_data() for multiple files, then returns the mean and 
standard deviation of each file

Created on Wed May 10 20:10:03 2023

@author: Branden Aitken
"""

def order(a):
    return int(a.split('Run')[1].split('_')[0])

def get_data(file,category):
    
    ind = {'ToA':-2, 'LG':-4,'HG':-3, 'ToT':-1}[category]
    
    data = {4:[],22:[]}
    
    with open(file) as f:
        
        for i,line in enumerate(f):
            
            #Takes out header
            if i >= 9:
                
                info = line.split()
                chan = int(info[-5])
                
                #Sometimes ToA is not collected
                #This Catches Those
                try:
                    pt = int(info[ind])
                    
                    if ind > -3: #Because of structure
                        pt = pt / 2
                        
                    data[chan].append(pt)
                    
                except ValueError:
                    pass #print(f'Line {i} Excluded From Sample')
               
    return np.array(data[4]),np.array(data[22])


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


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


# In[38]:


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


# In[39]:


def get_Counts_ToA_plot(wildcard, exclude, file_out, cable,gain = 'LG'):
    '''
    Plots the average ToA for each file found in the wildcard against
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


# In[40]:


file_no = glob.glob('/eos/user/s/salshama/MATHUSLA/Fibre_Bending_Test/No_bending/Run*_list.txt')
file_20 = glob.glob('/eos/user/s/salshama/MATHUSLA/Fibre_Bending_Test/20cm_bending/Run*_list.txt')
file_40 = glob.glob('/eos/user/s/salshama/MATHUSLA/Fibre_Bending_Test/40cm_bending/Run*_list.txt')
file_60 = glob.glob('/eos/user/s/salshama/MATHUSLA/Fibre_Bending_Test/60cm_bending/Run*_list.txt')


# In[41]:


#uplims=True,lolims=True,


# In[61]:


# CH4 LG

entries_4nolg, edges_4nolg, _ = plt.hist(ch4_lg_no,bins=np.linspace(70,300,100),histtype='step',label='0cm')
bin_centers_4nolg = 0.5*(edges_4nolg[:-1]+edges_4nolg[1:])
#plt.errorbar(bin_centers_4nolg,entries_4nolg,yerr=np.sqrt(entries_4nolg))#capsize=2,uplims=True,lolims=True,fmt='r.')

entries_420lg, edges_420lg, _ = plt.hist(ch4_lg_20,bins=np.linspace(70,300,100),histtype='step',label='20cm')
bin_centers_420lg = 0.5*(edges_420lg[:-1]+edges_420lg[1:])
#plt.errorbar(bin_centers_420lg,entries_420lg,yerr=np.sqrt(entries_420lg))#capsize=2,uplims=True,lolims=True,fmt='r.')

entries_440lg, edges_440lg, _ = plt.hist(ch4_lg_40,bins=np.linspace(70,300,100),histtype='step',label='40cm')
bin_centers_440lg = 0.5*(edges_440lg[:-1]+edges_440lg[1:])
#plt.errorbar(bin_centers_440lg,entries_440lg,yerr=np.sqrt(entries_440lg))#capsize=2,uplims=True,lolims=True,fmt='r.')

entries_460lg, edges_460lg, _ = plt.hist(ch4_lg_60,bins=np.linspace(70,300,100),histtype='step',label='60cm')
bin_centers_460lg = 0.5*(edges_460lg[:-1]+edges_460lg[1:])
#plt.errorbar(bin_centers_460lg,entries_460lg,yerr=np.sqrt(entries_460lg))#capsize=2,uplims=True,lolims=True,fmt='r.')

plt.title("Different Bending Radii (CH4, LG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('bending_radii_4lg')
plt.show()


# In[87]:


# CH4 HG

entries_4nohg, edges_4nohg, _ = plt.hist(ch4_hg_no,bins=np.linspace(70,2000,100),histtype='step',label='0cm')
bin_centers_4nohg = 0.5*(edges_4nohg[:-1]+edges_4nohg[1:])
#plt.errorbar(bin_centers_4nohg,entries_4nohg,yerr=np.sqrt(entries_4nohg))#,capsize=2,uplims=True,lolims=True,fmt='r.')

entries_420hg, edges_420hg, _ = plt.hist(ch4_hg_20,bins=np.linspace(70,1200,100),histtype='step',label='20cm')
bin_centers_420hg = 0.5*(edges_420hg[:-1]+edges_420hg[1:])
#plt.errorbar(bin_centers_420hg,entries_420hg,yerr=np.sqrt(entries_420hg))#,capsize=2,uplims=True,lolims=True,fmt='r.')

entries_440hg, edges_440hg, _ = plt.hist(ch4_hg_40,bins=np.linspace(70,800,100),histtype='step',label='40cm')
bin_centers_440hg = 0.5*(edges_440hg[:-1]+edges_440hg[1:])
#plt.errorbar(bin_centers_440hg,entries_440hg,yerr=np.sqrt(entries_440hg))#,capsize=2,uplims=True,lolims=True,fmt='r.')

entries_460hg, edges_460hg, _ = plt.hist(ch4_hg_60,bins=np.linspace(70,800,100),histtype='step',label='60cm')
bin_centers_460hg = 0.5*(edges_460hg[:-1]+edges_460hg[1:])
#plt.errorbar(bin_centers_460hg,entries_460hg,yerr=np.sqrt(entries_460hg))#,capsize=2,uplims=True,lolims=True,fmt='r.')

plt.title("Different Bending Radii (CH4, HG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('bending_radii_4hg')
plt.show()


# In[88]:


# CH22 LG

entries_22nolg, edges_22nolg, _ = plt.hist(ch22_lg_no,bins=np.linspace(70,1500,50),histtype='step',label='0cm')
bin_centers_22nolg = 0.5*(edges_22nolg[:-1]+edges_22nolg[1:])
#plt.errorbar(bin_centers_22nolg,entries_22nolg,yerr=np.sqrt(entries_22nolg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2220lg, edges_2220lg, _ = plt.hist(ch22_lg_20,histtype='step',label='20cm')
bin_centers_2220lg = 0.5*(edges_2220lg[:-1]+edges_2220lg[1:])
#plt.errorbar(bin_centers_2220lg,entries_2220lg,yerr=np.sqrt(entries_2220lg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2240lg, edges_2240lg, _ = plt.hist(ch22_lg_40,histtype='step',label='40cm')
bin_centers_2240lg = 0.5*(edges_2240lg[:-1]+edges_2240lg[1:])
#plt.errorbar(bin_centers_2240lg,entries_2240lg,yerr=np.sqrt(entries_2240lg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2260lg, edges_2260lg, _ = plt.hist(ch22_lg_60,histtype='step',label='60cm')
bin_centers_2260lg = 0.5*(edges_2260lg[:-1]+edges_2260lg[1:])
#plt.errorbar(bin_centers_2260lg,entries_2260lg,yerr=np.sqrt(entries_2260lg),capsize=2,uplims=True,lolims=True,fmt='r.')

plt.title("Different Bending Radii (CH22, LG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('bending_radii_22lg')
plt.show()


# In[94]:


#ch22_hg_60


# In[95]:


# CH22 HG

entries_22nohg, edges_22nohg, _ = plt.hist(ch22_hg_no,histtype='step',label='0cm')
bin_centers_22nohg = 0.5*(edges_22nohg[:-1]+edges_22nohg[1:])
#plt.errorbar(bin_centers_22nohg,entries_22nohg,yerr=np.sqrt(entries_22nohg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2220hg, edges_2220hg, _ = plt.hist(ch22_hg_20,histtype='step',label='20cm')
bin_centers_2220hg = 0.5*(edges_2220hg[:-1]+edges_2220hg[1:])
#plt.errorbar(bin_centers_2220hg,entries_2220hg,yerr=np.sqrt(entries_2220hg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2240hg, edges_2240hg, _ = plt.hist(ch22_hg_40,histtype='step',label='40cm')
bin_centers_2240hg = 0.5*(edges_2240hg[:-1]+edges_2240hg[1:])
#plt.errorbar(bin_centers_2240hg,entries_2240hg,yerr=np.sqrt(entries_2240hg),capsize=2,uplims=True,lolims=True,fmt='r.')

entries_2260hg, edges_2260hg, _ = plt.hist(ch22_hg_60,histtype='step',label='60cm')
bin_centers_2260hg = 0.5*(edges_2260hg[:-1]+edges_2260hg[1:])
#plt.errorbar(bin_centers_2260hg,entries_2260hg,yerr=np.sqrt(entries_2260hg),capsize=2,uplims=True,lolims=True,fmt='r.')

plt.title("Different Bending Radii (CH22, HG)")
plt.xlabel('Gain')
plt.ylabel('Counts')
plt.tight_layout()
plt.legend()
plt.savefig('bending_radii_22hg')
plt.show()


# In[42]:


# no bending radius

ch4_lg_no  = []
ch4_hg_no  = []
ch22_lg_no = []
ch22_hg_no = []

for file in file_no:
    ch4_lg_no  += list(get_data(file,'LG')[0])
    ch4_hg_no  += list(get_data(file,'HG')[0])
    ch22_lg_no += list(get_data(file,'LG')[1])
    ch22_hg_no += list(get_data(file,'HG')[1])
    
entries_4nolg, edges_4nolg, _ = plt.hist(ch4_lg_no,histtype='step',label='CH4 LG')
entries_4nohg, edges_4nohg, _ = plt.hist(ch4_hg_no,histtype='step',label='CH4 HG')

bin_centers_4nolg = 0.5*(edges_4nolg[:-1]+edges_4nolg[1:])
plt.errorbar(bin_centers_4nolg,entries_4nolg,yerr=np.sqrt(entries_4nolg),uplims=True,lolims=True,fmt='r.')

bin_centers_4nohg = 0.5*(edges_4nohg[:-1]+edges_4nohg[1:])
plt.errorbar(bin_centers_4nohg,entries_4nohg,yerr=np.sqrt(entries_4nohg),uplims=True,lolims=True,fmt='r.')

plt.title('No Bending Diameter - Ch4')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(70,105)
plt.tight_layout()
plt.legend()
plt.savefig('no_bending_ch4')
plt.show()

entries_22nolg, edges_22nolg, _ = plt.hist(ch22_lg_no,histtype='step',label='CH22 LG')
entries_22nohg, edges_22nohg, _ = plt.hist(ch22_hg_no,histtype='step',label='CH22 HG')

bin_centers_22nolg = 0.5*(edges_22nolg[:-1]+edges_22nolg[1:])
plt.errorbar(bin_centers_22nolg,entries_22nolg,yerr=np.sqrt(entries_22nolg),uplims=True,lolims=True,fmt='r.')

bin_centers_22nohg = 0.5*(edges_22nohg[:-1]+edges_22nohg[1:])
plt.errorbar(bin_centers_22nohg,entries_22nohg,yerr=np.sqrt(entries_22nohg),uplims=True,lolims=True,fmt='r.')

plt.title('No Bending Diameter - Ch22')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(70,105)
plt.tight_layout()
plt.legend(loc='center')
plt.savefig('no_bending_ch22')
plt.show()


# In[43]:


# 20cm bending radius

ch4_lg_20  = []
ch4_hg_20  = []
ch22_lg_20 = []
ch22_hg_20 = []

for file in file_20:
    ch4_lg_20  += list(get_data(file,'LG')[0])
    ch4_hg_20  += list(get_data(file,'HG')[0])
    ch22_lg_20 += list(get_data(file,'LG')[1])
    ch22_hg_20 += list(get_data(file,'HG')[1])
    
entries_420lg, edges_420lg, _ = plt.hist(ch4_lg_20,histtype='step',label='CH4 LG')

bin_centers_420lg = 0.5*(edges_420lg[:-1]+edges_420lg[1:])
plt.errorbar(bin_centers_420lg,entries_420lg,yerr=np.sqrt(entries_420lg),uplims=True,lolims=True,fmt='r.')

entries_420hg, edges_420hg, _ = plt.hist(ch4_hg_20,histtype='step',label='CH4 HG')

bin_centers_420lg = 0.5*(edges_420lg[:-1]+edges_420lg[1:])
plt.errorbar(bin_centers_420lg,entries_420lg,yerr=np.sqrt(entries_420lg),uplims=True,lolims=True,fmt='r.')

bin_centers_420hg = 0.5*(edges_420hg[:-1]+edges_420hg[1:])
plt.errorbar(bin_centers_420hg,entries_420hg,yerr=np.sqrt(entries_420hg),uplims=True,lolims=True,fmt='r.')

plt.title('20cm Bending Diameter - Ch4 (Lab Radius)')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(85,121)
plt.tight_layout()
plt.legend()
plt.savefig('20cm_bending_ch4')
plt.show()

entries_2220lg, edges_2220lg, _ = plt.hist(ch22_lg_20,histtype='step',label='CH22 LG')
entries_2220hg, edges_2220hg, _ = plt.hist(ch22_hg_20,histtype='step',label='CH22 HG')

bin_centers_2220lg = 0.5*(edges_2220lg[:-1]+edges_2220lg[1:])
plt.errorbar(bin_centers_2220lg,entries_2220lg,yerr=np.sqrt(entries_2220lg),uplims=True,lolims=True,fmt='r.')

bin_centers_2220hg = 0.5*(edges_2220hg[:-1]+edges_2220hg[1:])
plt.errorbar(bin_centers_2220hg,entries_2220hg,yerr=np.sqrt(entries_2220hg),uplims=True,lolims=True,fmt='r.')

plt.title('20cm Bending Diameter - Ch22 (Lab Radius)')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(85,121)
plt.tight_layout()
plt.legend()
plt.savefig('20cm_bending_ch22')
plt.show()


# In[44]:


# 40cm bending radius

ch4_lg_40  = []
ch4_hg_40  = []
ch22_lg_40 = []
ch22_hg_40 = []

for file in file_40:
    ch4_lg_40  += list(get_data(file,'LG')[0])
    ch4_hg_40  += list(get_data(file,'HG')[0])
    ch22_lg_40 += list(get_data(file,'LG')[1])
    ch22_hg_40 += list(get_data(file,'HG')[1])
    
entries_440lg, edges_440lg, _ = plt.hist(ch4_lg_40,histtype='step',label='CH4 LG')
entries_440hg, edges_440hg, _ = plt.hist(ch4_hg_40,histtype='step',label='CH4 HG')

bin_centers_440lg = 0.5*(edges_440lg[:-1]+edges_440lg[1:])
plt.errorbar(bin_centers_440lg,entries_440lg,yerr=np.sqrt(entries_440lg),uplims=True,lolims=True,fmt='r.')

bin_centers_440hg = 0.5*(edges_440hg[:-1]+edges_440hg[1:])
plt.errorbar(bin_centers_440hg,entries_440hg,yerr=np.sqrt(entries_440hg),uplims=True,lolims=True,fmt='r.')

plt.title('40cm Bending Diameter - Ch4')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(85,121)
plt.tight_layout()
plt.legend()
plt.savefig('40cm_bending_ch4')
plt.show()

entries_2240lg, edges_2240lg, _ = plt.hist(ch22_lg_40,histtype='step',label='CH22 LG')
entries_2240hg, edges_2240hg, _ = plt.hist(ch22_hg_40,histtype='step',label='CH22 HG')

bin_centers_2240lg = 0.5*(edges_2240lg[:-1]+edges_2240lg[1:])
plt.errorbar(bin_centers_2240lg,entries_2240lg,yerr=np.sqrt(entries_2240lg),uplims=True,lolims=True,fmt='r.')

bin_centers_2240hg = 0.5*(edges_2240hg[:-1]+edges_2240hg[1:])
plt.errorbar(bin_centers_2240hg,entries_2240hg,yerr=np.sqrt(entries_2240hg),uplims=True,lolims=True,fmt='r.')

plt.title('40cm Bending Diameter - Ch22')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(85,121)
plt.tight_layout()
plt.legend()
plt.savefig('40cm_bending_ch22')
plt.show()


# In[45]:


# 60cm bending radius

ch4_lg_60  = []
ch4_hg_60  = []
ch22_lg_60 = []
ch22_hg_60 = []

for file in file_60:
    ch4_lg_60  += list(get_data(file,'LG')[0])
    ch4_hg_60  += list(get_data(file,'HG')[0])
    ch22_lg_60 += list(get_data(file,'LG')[1])
    ch22_hg_60 += list(get_data(file,'HG')[1])
    
entries_460lg, edges_460lg, _ = plt.hist(ch4_lg_60,histtype='step',label='CH4 LG')
entries_460hg, edges_460hg, _ = plt.hist(ch4_hg_60,histtype='step',label='CH4 HG')

bin_centers_460lg = 0.5*(edges_460lg[:-1]+edges_460lg[1:])
plt.errorbar(bin_centers_460lg,entries_460lg,yerr=np.sqrt(entries_460lg),uplims=True,lolims=True,fmt='r.')

bin_centers_460hg = 0.5*(edges_460hg[:-1]+edges_460hg[1:])
plt.errorbar(bin_centers_460hg,entries_460hg,yerr=np.sqrt(entries_460hg),uplims=True,lolims=True,fmt='r.')

plt.title('60cm Bending Diameter - Ch4')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(70,105)
plt.tight_layout()
plt.legend()
plt.savefig('60cm_bending_ch4')
plt.show()

entries_2260lg, edges_2260lg, _ = plt.hist(ch22_lg_60,histtype='step',label='CH22 LG')
entries_2260hg, edges_2260hg, _ = plt.hist(ch22_hg_60,histtype='step',label='CH22 HG')

bin_centers_2260lg = 0.5*(edges_2260lg[:-1]+edges_2260lg[1:])
plt.errorbar(bin_centers_2260lg,entries_2260lg,yerr=np.sqrt(entries_2260lg),uplims=True,lolims=True,fmt='r.')

bin_centers_2260hg = 0.5*(edges_2260hg[:-1]+edges_2260hg[1:])
plt.errorbar(bin_centers_2260hg,entries_2260hg,yerr=np.sqrt(entries_2260hg),uplims=True,lolims=True,fmt='r.')

plt.title('60cm Bending Diameter - Ch22')
plt.ylabel('Counts')
plt.xlabel('Gain')
#plt.xlim(70,105)
plt.tight_layout()
plt.legend()
plt.savefig('60cm_bending_ch22')
plt.show()


# In[46]:


print("(Ch4 LG, no) Mean is: ",np.mean(ch4_lg_no))
print("(Ch4 LG, no) Standard deviation is: ",np.std(ch4_lg_no))

print("(Ch4 HG, no) Mean is: ",np.mean(ch4_hg_no))
print("(Ch4 HG, no) Standard deviation is: ",np.std(ch4_hg_no))

print("(Ch22 LG, no) Mean is: ",np.mean(ch22_lg_no))
print("(Ch22 LG, no) Standard deviation is: ",np.std(ch4_lg_no))

print("(Ch22 HG, no) Mean is: ",np.mean(ch22_hg_no))
print("(Ch22 HG, no) Standard deviation is: ",np.std(ch4_hg_no))


# In[47]:


print("(Ch4 LG, 20cm) Mean is: ",np.mean(ch4_lg_20))
print("(Ch4 LG, 20cm) Stdev is: ",np.std(ch4_lg_20),"\n")

print("(Ch4 HG, 20cm) Mean is: ",np.mean(ch4_hg_20))
print("(Ch4 HG, 20cm) Stdev is: ",np.std(ch4_hg_20),"\n")

print("(Ch22 LG, 20cm) Mean is: ",np.mean(ch22_lg_20))
print("(Ch22 LG, 20cm) Stdev is: ",np.std(ch4_lg_20),"\n")

print("(Ch22 HG, 20cm) Mean is: ",np.mean(ch22_hg_20))
print("(Ch22 HG, 20cm) Stdev is: ",np.std(ch4_hg_20),"\n")


# In[48]:


print("(Ch4 LG, 40cm) Mean is: ",np.mean(ch4_lg_40))
print("(Ch4 LG, 40cm) Stdev is: ",np.std(ch4_lg_40),"\n")

print("(Ch4 HG, 40cm) Mean is: ",np.mean(ch4_hg_40))
print("(Ch4 HG, 40cm) Stdev is: ",np.std(ch4_hg_40),"\n")

print("(Ch22 LG, 40cm) Mean is: ",np.mean(ch22_lg_40))
print("(Ch22 LG, 40cm) Stdev is: ",np.std(ch4_lg_40),"\n")

print("(Ch22 HG, 40cm) Mean is: ",np.mean(ch22_hg_40))
print("(Ch22 HG, 40cm) Stdev is: ",np.std(ch4_hg_40))


# In[ ]:


print("(Ch4 LG, 60cm) Mean is: ",np.mean(ch4_lg_60))
print("(Ch4 LG, 60cm) Stdev is: ",np.std(ch4_lg_60),"\n")

print("(Ch4 HG, 60cm) Mean is: ",np.mean(ch4_hg_60))
print("(Ch4 HG, 60cm) Stdev is: ",np.std(ch4_hg_60),"\n")

print("(Ch22 LG, 60cm) Mean is: ",np.mean(ch22_lg_60))
print("(Ch22 LG, 60cm) Stdev is: ",np.std(ch4_lg_60),"\n")

print("(Ch22 HG, 60cm) Mean is: ",np.mean(ch22_hg_60))
print("(Ch22 HG, 60cm) Stdev is: ",np.std(ch4_hg_60))


# In[ ]:




