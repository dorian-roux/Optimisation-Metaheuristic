##################################################
#  Optimization Metaheuristic - Data Generation  #
##################################################

# - LIBRARIES - 
import os
import json
import random
import numpy as np
import pandas as pd


# - FUNCTIONS - 

# -- FOG - Construct VMS 
def setFOG_VMS():
    dictVMS = {
        'f-tiny': {'MIPS_pCPU': 1, 'CPU': 0.5, 'BANDWITH': 0.1}, 
        'f-xsmall': {'MIPS_pCPU': 1, 'CPU': 1, 'BANDWITH': 0.1}, 
        'f-small': {'MIPS_pCPU': 1, 'CPU': 2, 'BANDWITH': 0.1},
        'f-large': {'MIPS_pCPU': 2, 'CPU': 2, 'BANDWITH': 0.5},
        'f-xlarge': {'MIPS_pCPU': 6, 'CPU': 1, 'BANDWITH': 1},
        'f-xxlarge': {'MIPS_pCPU': 4, 'CPU': 2, 'BANDWITH': 1}
    }

    pathData = 'data/FOG'
    os.makedirs(pathData, exist_ok=True)
    with open(os.path.join(pathData, 'FOG-VMS-Catalog.json'), 'w') as file:
        json.dump(dictVMS, file, indent=2)    
    return dictVMS



# -- CLOUD - Construct VMS 
def setCLOUD_VMS():
    dictVMS = {
        'c-xsmall': {'MIPS_pCPU': 2, 'CPU': 1, 'BANDWITH': 1}, 
        'c-small': {'MIPS_pCPU': 6, 'CPU': 1, 'BANDWITH': 2},
        'c-large': {'MIPS_pCPU': 6, 'CPU': 2, 'BANDWITH': 5},
        'c-xlarge': {'MIPS_pCPU': 8, 'CPU': 2, 'BANDWITH': 10},
    }
    
    pathData = 'data/CLOUD'
    os.makedirs(pathData, exist_ok=True)
    with open(os.path.join(pathData, 'CLOUD-VMS-Catalog.json'), 'w') as file:
        json.dump(dictVMS, file, indent=2)    
    return dictVMS



# -- FOG → Nodes & VMs --
def setEnvironment_FOG(n_Nodes = 5, range_Distance = ((0, 50), (0, 50)), range_VMS = (2, 6)):
    """
    Summary:
        We are looking to randomly generate FOG Nodes that include the following INITIAL information:
        - the Node ID
        - the Node X-Coordinate of a Cartésian-Plan between [0-50, 0-50]km
        - the Node Y-Coordinate of a Cartésian-Plan between [0-50, 0-50]km
        - the Node Power between [0.5, 1]KWatt
        - the number of VMs associated to the Node within [2, 6]

        Among each Node we define the VMs that include the following INITIAL information:
        - the Node ID
        - the VM ID
        - the VM Compute Power within [1, 8] MIpS (Million Instructions per Second)
        - the VM Bandwith [0.1, 1] GbpS
    
    Args:
        n_Nodes (int, optional): Number of FOG Nodes that will be generated. Defaults to 5.
        range_DISTANCE (tuple, optional): tuple containing a couple of tuples with respectively the range within the X and Y Coordinates. Defaults to ((0, 50), (0, 50)).
        range_VMS (tuple, optional): Range between the Minimum and Maximum VMS per FOG Node. Defaults to (2, 6).
    """    
    
    # Set Variables
    dictVMS = setFOG_VMS()
    min_VMS_perES, max_VMS_perES = range_VMS
    rangeX, rangeY = range_Distance
    
    # Set FOG Nodes dataframe
    nodesFOG = pd.DataFrame()
    nodesFOG['NODE_ID'] = list(map(lambda n_Node : f'F{"0" * (len(str(n_Nodes)) - len(str(n_Node)))}{n_Node}' if n_Node > 10 else f'F0{n_Node}', np.arange(1, n_Nodes+1)))
    nodesFOG['X'] = np.random.uniform(rangeX[0], rangeX[1], n_Nodes).round(2)
    nodesFOG['Y'] = np.random.uniform(rangeY[0], rangeY[0], n_Nodes).round(2)
    nodesFOG['POWER'] = np.random.uniform(0.5, 1, n_Nodes).round(2)
    nodesFOG['TOTAL_MEMORY'] = np.random.randint(1, 10, n_Nodes)
    nodesFOG['N_VMS'] = np.random.randint(min_VMS_perES, max_VMS_perES, n_Nodes)


    # Set VMs dataframe
    nodesVMs = pd.DataFrame()
    for _,rowNode in nodesFOG.iterrows():
        for n in range(rowNode.N_VMS):
            randVM_INSTANCE = random.choice(list(dictVMS.keys()))
            randVM = dictVMS[randVM_INSTANCE]
            nodesVMs = pd.concat([nodesVMs, pd.DataFrame({'NODE_ID': rowNode.NODE_ID, 
                                                    'VM_ID': f'{rowNode.NODE_ID}_' + (f'VM{"0" * (len(str(rowNode.N_VMS)) - len(str(n)))}{n+1}' if n > 10 else f'VM0{n+1}'), 
                                                    'VM_INSTANCE': randVM_INSTANCE,
                                                    'VM_POWER_CAPACITY': round(randVM['CPU'] * randVM['MIPS_pCPU']),
                                                    'VM_CPU': randVM['CPU'],
                                                    'VM_POWER_CAPACITY_PER_CPU': randVM['MIPS_pCPU'],
                                                    'VM_BANDWITH': randVM['BANDWITH']}, index=[0])])
    nodesFOG = nodesFOG.merge(nodesVMs.groupby('NODE_ID')['VM_POWER_CAPACITY'].sum().reset_index(name='TOTAL_POWER_CAPACITY'), on='NODE_ID')

    # Save both DATAFRAME as a CSV File
    pathData = 'data/FOG'
    os.makedirs(pathData, exist_ok=True)
    nodesFOG.to_csv(os.path.join(pathData, 'FOG-Nodes.csv'), index=0)
    nodesVMs.to_csv(os.path.join(pathData, 'FOG-Nodes-VMS.csv'), index=0)
    
    
    return nodesFOG, nodesVMs


# -- CLOUD → Nodes & VMs --
def setEnvironment_CLOUD(n_Nodes = 2, range_Distance = ((100, 500), (100, 500)), range_VMS = (1, 3)):
    """
    Summary:
        We are looking to randomly generate FOG Nodes that include the following INITIAL information:
        - the Node ID
        - the Node X-Coordinate of a Cartésian-Plan between [100-500, 100-500]km
        - the Node Y-Coordinate of a Cartésian-Plan between [100-500, 100-500]km
        - the Node Power between [1, 2.5]KWatt
        - the number of VMs associated to the Node within [1, 3]

        Among each Node we define the VMs that include the following INITIAL information:
        - the Node ID
        - the VM ID
        - the VM Compute Power within [2, 16] MIpS (Million Instructions per Second)
        - the VM Bandwith [1, 10] GbpS
    
    Args:
        n_Nodes (int, optional): Number of CLOUD Nodes that will be generated. Defaults to 2.
        range_DISTANCE (tuple, optional): tuple containing a couple of tuples with respectively the range within the X and Y Coordinates. Defaults to ((100, 500), (100, 500)).
        range_VMS (tuple, optional): Range between the Minimum and Maximum VMS per CLOUD Node. Defaults to (1, 3).
    """    
    
    # Set Variables
    dictVMS = setCLOUD_VMS()
    min_VMS_perES, max_VMS_perES = range_VMS
    rangeX, rangeY = range_Distance
    
    # Set CLOUD Nodes dataframe
    nodesCLOUD = pd.DataFrame()
    nodesCLOUD['NODE_ID'] = list(map(lambda n_Node : f'C{"0" * (len(str(n_Nodes)) - len(str(n_Node)))}{n_Node}' if n_Node > 10 else f'C0{n_Node}', np.arange(1, n_Nodes+1)))
    nodesCLOUD['X'] = np.random.uniform(rangeX[0], rangeX[1], n_Nodes).round(2)
    nodesCLOUD['Y'] = np.random.uniform(rangeY[0], rangeY[0], n_Nodes).round(2)
    nodesCLOUD['POWER'] = np.random.uniform(1, 2.5, n_Nodes).round(2)
    nodesCLOUD['N_VMS'] = np.random.randint(min_VMS_perES, max_VMS_perES, n_Nodes)


    # Set VMs dataframe
    nodesVMs = pd.DataFrame()
    for _,rowNode in nodesCLOUD.iterrows():
        for n in range(rowNode.N_VMS):
            randVM_INSTANCE = random.choice(list(dictVMS.keys()))
            randVM = dictVMS[randVM_INSTANCE]
            nodesVMs = pd.concat([nodesVMs, pd.DataFrame({'NODE_ID': rowNode.NODE_ID, 
                                                    'VM_ID': f'{rowNode.NODE_ID}_' + (f'VM{"0" * (len(str(rowNode.N_VMS)) - len(str(n)))}{n+1}' if n > 10 else f'VM0{n+1}'), 
                                                    'VM_INSTANCE': randVM_INSTANCE,
                                                    'VM_POWER_CAPACITY': round(randVM['CPU'] * randVM['MIPS_pCPU']),
                                                    'VM_CPU': randVM['CPU'],
                                                    'VM_POWER_CAPACITY_PER_CPU': randVM['MIPS_pCPU'],
                                                    'VM_BANDWITH': randVM['BANDWITH']}, index=[0])])
    nodesCLOUD = nodesCLOUD.merge(nodesVMs.groupby('NODE_ID')['VM_POWER_CAPACITY'].sum().reset_index(name='TOTAL_POWER_CAPACITY'), on='NODE_ID')

    # Save both DATAFRAME as a CSV File
    pathData = 'data/CLOUD'
    os.makedirs(pathData, exist_ok=True)
    nodesCLOUD.to_csv(os.path.join(pathData, 'CLOUD-Nodes.csv'), index=0)
    nodesVMs.to_csv(os.path.join(pathData, 'CLOUD-Nodes-VMS.csv'), index=0)
    
    return nodesCLOUD, nodesVMs



# -- IoT → Objects & Tasks --
def setEnvironment_IoT_n_TASKS(n_IoTs = 20, range_Distance = ((0, 50), (0, 50))):
    """
    Summary:
        We are looking to generate IoT objects and their associated TASK that include the following INITIAL information:
        - the IoT ID
        - the IoT X-Coordinate of a Cartésian-Plan between [0-50, 0-50]km
        - the IoT Y-Coordinate of a Cartésian-Plan between [0-50, 0-50]km
        - the IoT Tasks Number (We set this variable as 1 which mean that each IoT object has exactly 1 Task)
        - the IoT Compute Power within [0.15, 0.75] MIpS (Million Instructions per Second)
        - the IoT Power between [0.050, 0.125]KWatt

        Among each IoT we define the Task(s) that include the following INITIAL information:
        - the IoT ID
        - the Task ID
        - the Task MIpS [1, 50] (Million Instructions per Second)
        - the Task File Size [1, 200]MO (MegaOctets)
        - the required memory
    
     Args:
        n_IoTs (int, optional): Number of IoT Objects that will be generated. Defaults to 20.
        range_DISTANCE (tuple, optional): tuple containing a couple of tuples with respectively the range within the X and Y Coordinates. Defaults to ((0, 50), (0, 50)).
    """
    
    # Set Variables
    rangeX, rangeY = range_Distance
    
    # Set IoT Objects dataframe
    baseIoT = pd.DataFrame()
    baseIoT['IoT_ID'] = list(map(lambda n_IoT : f'I{"0" * (len(str(n_IoTs)) - len(str(n_IoT)))}{n_IoT}' if n_IoT > 10 else f'I0{n_IoT}', np.arange(1, n_IoTs+1)))
    baseIoT['X'] = np.random.uniform(rangeX[0], rangeX[1], n_IoTs).round(2)
    baseIoT['Y'] = np.random.uniform(rangeY[0], rangeY[1], n_IoTs).round(2)
    baseIoT['POWER'] = np.random.uniform(0.05, 0.125, n_IoTs).round(4)
    baseIoT['IoT_MIpS'] = np.random.uniform(0.15, 0.75, n_IoTs).round(4)
    baseIoT['N_TASKS'] = 1

    # Set TASKS dataframe
    baseTASK = pd.DataFrame()
    for _,rowIoT in baseIoT.iterrows():
        for n in range(rowIoT.N_TASKS):
            baseTASK = pd.concat([baseTASK, pd.DataFrame({'IoT_ID': rowIoT.IoT_ID, 
                                                    'TASK_ID': f'{rowIoT.IoT_ID}_' + (f'T{"0" * (len(str(rowIoT.N_VMS)) - len(str(n)))}{n+1}' if n > 10 else f'T0{n+1}'), 
                                                    'N_INSTRUCTIONS': np.random.randint(1, 50),
                                                    'FILE_SIZE': np.random.randint(1, 200)}, index=[0])])

    # Save both DATAFRAME as a CSV File
    pathData = 'data/IoT'
    os.makedirs(pathData, exist_ok=True)
    baseIoT.to_csv(os.path.join(pathData, 'base-IoT.csv'), index=0)
    baseTASK.to_csv(os.path.join(pathData, 'base-Tasks.csv'), index=0)
    
    return baseIoT, baseTASK



# - CORE -
if __name__ == '__main__':
    setEnvironment_FOG()
    setEnvironment_CLOUD()
    setEnvironment_IoT_n_TASKS()