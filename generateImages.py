'''Packages for data processing'''
import numpy as np
import pandas as pd



'''Packages for file processing'''
import os
import glob
import sys

'''Packages for plotting'''
import matplotlib.pyplot as plt

from sklearn import preprocessing as pp
from sklearn.preprocessing import minmax_scale

def generateIATs(path):
    df = pd.read_csv(path,index_col=False)
    df.reset_index(drop=True, inplace=True)
    unique_src_IPs = df['src_ip'].unique()
    iat_list = []
    for id in unique_src_IPs:
        ts_id = df.loc[df['src_ip']==id,'timestamp'] 
        iats_id = np.asarray(np.array(ts_id.diff().fillna(ts_id))) #See: https://codereview.stackexchange.com/questions/210070/calculating-time-deltas-between-rows-in-a-pandas-dataframe
        iat_list.append(iats_id)
    return unique_src_IPs, iat_list

def generateImages(data,unique_src_IPs):
    #get the min and max
    min_list = []
    max_list = []
    for i in range(len(unique_src_IPs)):
        print(i)
        id = data[i].tolist()
        #print(id)
        min_list.append(min(id))
        max_list.append(max(id))
    y_min = min(min_list)
    y_max = max(max_list)

    n=100
    for i in range(len(unique_src_IPs)):
        print(i)
        id = data[i].tolist()
        chunked_id = [id[i * n:(i + 1) * n] for i in range((len(id) + n - 1) // n )]
        plotAndSave(chunked_id,y_min,y_max,unique_src_IPs[i])

    return 1

def plotAndSave(data,y_min,y_max,src_IP):
    x_data=range(0,100)
    i = 0
    for y_data in data:
        if(len(y_data))<len(x_data):
            x_data=range(0,len(y_data))
        plt.plot(x_data,y_data)
        #plt.ylim(y_min, y_max)
        #plt.xlim(1,100)
        plt.savefig(r'C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\iat_new\iat_'+str(src_IP)+"_"+str(i)+".png")
        plt.show()        
        i+=1


'''Paths to the input files'''
path1= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151020_transport.csv"
path2= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151021_transport.csv"
path3= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022_transport.csv"

'''IAT Graph Images'''
unique_src_IPs, iat_list = generateIATs(path1)
#minmax_scale(iat_list)
#mm_scaler = pp.MinMaxScaler()
#iat_list_scaled = mm_scaler.fit_transform(iat_list)
#print(iat_list)
#print(iat_list_scaled)
generateImages(iat_list,unique_src_IPs)
print("DONE")
