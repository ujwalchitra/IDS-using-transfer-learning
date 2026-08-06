import os

# UT 2024 Dataset Configuration
UT2024_CONFIG = {
    'dataset_name': 'CIC-IDS-2018',
    'folder_path': 'UT_2024/',
    'files': [
        'Botnet-Friday-02-03-2018.csv',
        'Bruteforce-Wednesday-14-02-2018.csv', 
        'DDoS1-Tuesday-20-02-2018.csv',
        'DDoS2-Wednesday-21-02-2018.csv',
        'DoS1-Thursday-15-02-2018.csv',
        'DoS2-Friday-16-02-2018.csv',
        'Infil1-Wednesday-28-02-2018.csv',
        'Infil2-Thursday-01-03-2018.csv',
        'Web1-Thursday-22-02-2018.csv',
        'Web2-Friday-23-02-2018.csv'
    ],
    'attack_types': {
        'Botnet': ['Botnet-Friday-02-03-2018.csv'],
        'Bruteforce': ['Bruteforce-Wednesday-14-02-2018.csv'],
        'DDoS': ['DDoS1-Tuesday-20-02-2018.csv', 'DDoS2-Wednesday-21-02-2018.csv'],
        'DoS': ['DoS1-Thursday-15-02-2018.csv', 'DoS2-Friday-16-02-2018.csv'],
        'Infiltration': ['Infil1-Wednesday-28-02-2018.csv', 'Infil2-Thursday-01-03-2018.csv'],
        'Web_Attack': ['Web1-Thursday-22-02-2018.csv', 'Web2-Friday-23-02-2018.csv']
    },
    'features': {
        'network_flow': ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts'],
        'packet_stats': ['TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min'],
        'timing': ['Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min'],
        'flags': ['FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt']
    },
    'target_column': 'Label',
    'normal_label': 'Benign'
}