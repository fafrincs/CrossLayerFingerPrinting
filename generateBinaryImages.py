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
from PIL import Image


def groupBinDataByIPs(path):
    df = pd.read_csv(path,index_col=False)
    df.reset_index(drop=True, inplace=True)
    unique_src_IPs = df['src_ip'].unique()
    bin_list = []
    for id in unique_src_IPs:
        bin_id = df.loc[df['src_ip']==id,'bin_data'] 
        bin_list.append(bin_id)
    return unique_src_IPs, bin_list#format: bin_list[0]--->bin_data for src_IP[0]

def generateBinImages(data,unique_src_IPs):
    n=100
    for i in range(len(unique_src_IPs)):
        id = np.asarray(data[i])
        chunked_id = [id[i * n:(i + 1) * n] for i in range((len(id) + n - 1) // n )]
        # print(type(chunked_id[0][0]))#str--->0b10000000000000100000000111111.....
        bin_chunk_id = convertToBinaryVector(chunked_id)#type(bin_chunk_id)---> list of Numpy arrays
        bin_chunk_id = convertToRectangle(bin_chunk_id) 
        #print(type(bin_chunk_id[0][0]))  
        #print(bin_chunk_id[0][0])
        print(len(bin_chunk_id)) 
        plotAndSave(bin_chunk_id,unique_src_IPs[i])
        #break
    return 1


def convertToRectangle(bin_chunks):
    for i in range(len(bin_chunks)):
        #print("*********")
        max_len = max(map(len, bin_chunks[i]))
        min_len = min(map(len, bin_chunks[i]))
        #print(max_len)
        #print(min_len)
        pad_len = max_len-min_len
        padding = "0" * pad_len
        for j in range(len(bin_chunks[i])):
                #lst = list(bin_chunks[i][j])
                #lst.append(padding)
                lst = str(bin_chunks[i][j])
                lst=lst+padding
                bin_chunks[i][j] = np.array(lst)
                #bin_chunks[i][j] = int(np.array(bin_chunks[i][j]))
        #print(max(map(len, bin_chunks[i])))
        #print(min(map(len, bin_chunks[i])))
        #print("*********")
    #bin_chunks = np.concatenate( bin_chunks, axis=0 )
    return bin_chunks


def convertToBinaryVector(data_chunk):
    bin_chunk = []
    for i in range(len(data_chunk)):
        bin_vec = []
        for j in range(len(data_chunk[i])):
            bin_vec.append(np.array(data_chunk[i][j][2:]))
        bin_chunk.append(np.array(bin_vec))
    
    return bin_chunk



def plotAndSave(data,src_IP):

    i=0
    for mat in data:
        filename = r'C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\binary\bin_'+str(src_IP)+"_"+str(i)+".txt"
        mat = np.array(mat)
        #print(mat)
        i+=1
        np.savetxt(filename, mat, fmt='%s')
    

def saveAsBitStream(unique_src_IPs, bin_list):
        #bit_stream_id=[]
        for i in range(len(unique_src_IPs)):
            tmp=''
            id = np.asarray(bin_list[i])
            #print(id)
            for j in range(len(id)):
                #print(id[j])
                tmp = tmp+id[j][2:]
            #bit_stream_id.append(tmp)
            filename = r'C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\bit_stream'+str(unique_src_IPs[i])+".txt"
            with open(filename, 'w+') as fh:
                fh.write(tmp)
        #print(len(bit_stream_id))


            
                

    


'''Paths to the input files'''
path1= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151020.csv"
path2= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151021.csv"
path3= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022.csv"

'''Binary Images'''
unique_src_IPs, bin_list = groupBinDataByIPs(path1)
#print(bin_list)
# generateBinImages(bin_list,unique_src_IPs)
# print("DONE")

saveAsBitStream(unique_src_IPs, bin_list)
