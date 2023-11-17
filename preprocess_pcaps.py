import numpy as np
import pandas as pd
from scapy.all import *
import os
import glob
import sys

#import pyshark


# path = r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151020.pcap"
# capture = pyshark.FileCapture(path, keep_packets=True)

# for i, packet in enumerate(capture):
#         print(packet.layers)
#         print(packet.frame_info)
#         print(packet.frame_info.time_epoch)
#         print(packet)

# def filterByLayer(path):
#     for fname in os.listdir(path4SICS):
#         if fname.endswith('.pcap'):
#             # do stuff on the file

            # for packet in PcapReader('file.pcap')
            #     try:
            #         print(packet[IPv6].src)
            #     except:
            #         pass
            


# #read the data##
# path4SICS = r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics"

scapy_cap = rdpcap(r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151020.pcap")
#scapy_cap = rdpcap(r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022.pcap")

'''To create (src_ip, binary_payload) csv'''
src_ip=[]
binary_stream=[]
for pkt in scapy_cap:
    if IP in pkt:
        #print(pkt[IP].src)
        #print(bin(int.from_bytes(raw(pkt),byteorder=sys.byteorder)))
        #print("    ")
        src_ip.append(pkt[IP].src)
        binary_stream.append(bin(int.from_bytes(raw(pkt),byteorder=sys.byteorder)))
df = pd.DataFrame({'src_ip':src_ip,'bin_data':binary_stream})
df.to_csv(r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022.csv")





scapy_cap = rdpcap(r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022.pcap")
src_ip=[]
dst_ip=[]
time_stream_start=[]#TSval
time_stream_prev=[]#TSecr
timestamp = []
layers=[]
for pkt in scapy_cap:
    if (IP in pkt):
         src_ip.append(pkt[IP].src)
         dst_ip.append(pkt[IP].dst)
    #     # time_stream_start.append(pkt[TCP].Timestamp[0])
    #     # time_stream_prev.append(pkt[TCP].Timestamp[1])
         timestamp.append(pkt.time)
         layers.append(pkt.payload.layers())
df = pd.DataFrame({'src_ip':src_ip,'dst_ip':dst_ip, 'timestamp':timestamp, 'layers':layers})
df.to_csv(r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022_transport.csv")


# '''Unique IP addresses'''
# dic = {}
# for pkt in scapy_cap:
#     temp = pkt.sprintf("%IP.dst%")
#     temp = pkt.sprintf("%IP.src%")
#     dic[temp] = 1

# for ip in dic.keys():
#     print(ip)
