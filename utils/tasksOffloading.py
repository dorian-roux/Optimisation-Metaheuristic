#####################################################
#  Optimization Metaheuristic - Prepare Offloading  #
#####################################################

# - LIBRARIES - 

# -- General --
import os
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# -- Script based Packages --
from generateData import setEnvironment_FOG, setEnvironment_CLOUD, setEnvironment_IoT_n_TASKS

# - FUNCTIONS - 

# -- Merge FOG/CLOUD VMS --
def setBaseVMS(FOG_nNodes = 5, CLOUD_nNodes = 3, FOG_rDist = ((0,50), (0,50)), CLOUD_rDist = ((100, 500), (100, 500)), FOG_rVMS = (2, 6), CLOUD_rVMS = (2,4), saveInformation = False):
    base_FOG, base_FOG_VMS = setEnvironment_FOG(FOG_nNodes, FOG_rDist, FOG_rVMS)
    base_CLOUD, base_CLOUD_VMS = setEnvironment_CLOUD(CLOUD_nNodes, CLOUD_rDist, CLOUD_rVMS)  
    base_FOG_VMS = base_FOG.merge(base_FOG_VMS, on='NODE_ID')
    base_CLOUD_VMS = base_CLOUD.merge(base_CLOUD_VMS, on='NODE_ID')
    baseVMS = pd.concat([base_FOG_VMS[['NODE_ID', 'POWER', 'VM_POWER_CAPACITY', 'TOTAL_MEMORY', 'TOTAL_POWER_CAPACITY', 'X', 'Y', 'VM_ID', 'VM_BANDWITH']], base_CLOUD_VMS[['NODE_ID', 'POWER', 'VM_POWER_CAPACITY', 'TOTAL_POWER_CAPACITY', 'X', 'Y', 'VM_ID', 'VM_BANDWITH']]])
    baseVMS['QUEUE_TASK'] = 0
    baseVMS['QUEUE_TIME'] = 0
    
    if saveInformation:
        # Save DATAFRAME as a CSV File
        pathData = 'data'
        os.makedirs(pathData, exist_ok=True)
        baseVMS.to_csv(os.path.join(pathData, 'Bases-VMS.csv'), index=0)
    return baseVMS


# -- Compute Propagation Time --
def computePropagationTime(posIoT, posVM):

    # Set and Assign Variables 
    propagationSpeed = 2.25 * (10**8)
    obj_X, obj_Y, vm_X, vm_Y = posIoT[0], posIoT[1], posVM[0], posVM[1]

    # Compute Distance based on the Cartesian Coordinates
    dist_inKM = math.sqrt( (obj_X - vm_X)**2 + (obj_Y - vm_Y)**2 )
    dist_inM = dist_inKM * (10**3)

    # Propagation Time
    # Distance : in meters [m]
    # Speed : in meters per seconds [m/s] 
    return (dist_inM / propagationSpeed)  # Return a Propagation Time [in seconds]



# -- Compute Transmission Time --
def computeTransmissionTime(Task_FileSize, VM_Bandwith):
    
    # Set and Assign Variables 
    bit_to_octets = 1/8
    obj_bandwdith = (VM_Bandwith*bit_to_octets)*(10**3) # Gbps into MOpS

    # Transmission Time
    # Input Data Volume : in MegaBytes [MB] == Megaoctets [MO]
    # Bandwidth that connect an IoT object and a Virtual Machine : in Kilooctets per seconds [MOpS] 
    return Task_FileSize / obj_bandwdith




# -- Compute FOG Latency --
def compute_FOG_BestLatency(TaskInfo, baseVMS):
    baseFOG_VMS = baseVMS[baseVMS.VM_ID.str.contains('F')]
    bestLatency, saveTrmsnTime, saveIndex = 10**8, 10**8, 0, 
    for F_INDEX, F_VM in baseFOG_VMS.iterrows():
        QTime = 0 if F_VM.QUEUE_TIME is None else F_VM.QUEUE_TIME  # Queue Time
        PgtTime = computePropagationTime((TaskInfo.X, TaskInfo.Y), (F_VM.X, F_VM.Y)) # Propagation Time
        TrmsnTime = computeTransmissionTime(TaskInfo.FILE_SIZE, F_VM.VM_BANDWITH) # Transmission Time
        if (QTime + PgtTime + TrmsnTime) < bestLatency:
            saveIndex = F_INDEX
            bestLatency = (QTime + PgtTime + TrmsnTime)
            saveTrmsnTime = TrmsnTime
    return baseVMS.iloc[saveIndex], bestLatency, saveTrmsnTime


# -- Compute FOG Energy Consumption --
def computeFOG_EC(FOG_TrmsnTime, TaskInfo, best_FOG_VM):
    FOG_Energy_TrsmsnPhase = FOG_TrmsnTime * (TaskInfo.POWER * 1000)
    FOG_Energy_PrsngPhase =  (best_FOG_VM.POWER * 1000) * (TaskInfo.N_INSTRUCTIONS / best_FOG_VM.VM_POWER_CAPACITY)
    return FOG_Energy_TrsmsnPhase + FOG_Energy_PrsngPhase



# -- Compute CLOUD Latency
def compute_CLOUD_BestLatency(TaskInfo, baseVMS):
    baseCLOUD_VMS = baseVMS[baseVMS.VM_ID.str.contains('C')]
    bestLatency, saveTrmsnTime, saveIndex = 10**8, 10**8, 0, 
    for C_INDEX, C_VM in baseCLOUD_VMS.iterrows():
        QTime = 0 if C_VM.QUEUE_TIME is None else C_VM.QUEUE_TIME  # Queue Time
        PgtTime = computePropagationTime((TaskInfo.X, TaskInfo.Y), (C_VM.X, C_VM.Y)) # Propagation Time
        TrmsnTime = computeTransmissionTime(TaskInfo.FILE_SIZE, C_VM.VM_BANDWITH) # Transmission Time
        if (QTime + PgtTime + TrmsnTime) < bestLatency:
            saveIndex = C_INDEX
            bestLatency = (QTime + PgtTime + TrmsnTime)
            saveTrmsnTime = TrmsnTime
    return baseCLOUD_VMS.iloc[saveIndex], bestLatency, saveTrmsnTime


# -- Compute CLOUD Energy Consumption
def computeCLOUD_EC(FOG_TrmsnTime, CLOUD_TrmsnTime, TaskInfo, best_FOG_CLOUD):
    FOG_Energy_TrsmsnPhase = FOG_TrmsnTime * (TaskInfo.POWER * 1000)
    CLOUD_Energy_TrsmsnPhase = CLOUD_TrmsnTime * (TaskInfo.POWER * 1000)
    CLOUD_Energy_PrsngPhase =  (best_FOG_CLOUD.POWER * 1000) * (TaskInfo.N_INSTRUCTIONS / best_FOG_CLOUD.VM_POWER_CAPACITY)
    return FOG_Energy_TrsmsnPhase + CLOUD_Energy_TrsmsnPhase + CLOUD_Energy_PrsngPhase


  
# -- Prepare Offloading - V2
def prepareOffloading_V2(n_IoTs = 20, FOG_nNodes = 5, CLOUD_nNodes = 2, IoT_rDist=((0,50), (0,50)), FOG_rDist = ((0,50), (0,50)), CLOUD_rDist = ((100, 500), (100, 500)), FOG_rVMS = (2, 6), CLOUD_rVMS = (2,4), saveInformation=False):
    taskVM, taskVM_Dummy = pd.DataFrame(), pd.DataFrame()
    
    baseVMS = setBaseVMS(FOG_nNodes, CLOUD_nNodes, FOG_rDist, CLOUD_rDist, FOG_rVMS, CLOUD_rVMS)
    baseVMS['TYPE'] = np.where(baseVMS['NODE_ID'].str.contains('F'), 'FOG', 'CLOUD')
    baseVMS['MEMORY_REMAINING'] = baseVMS['TOTAL_MEMORY']

    baseIoT, baseTasks = setEnvironment_IoT_n_TASKS(n_IoTs, IoT_rDist)
    baseIoT = baseIoT.merge(baseTasks, on='IoT_ID')
    baseIoT['TASK_ASSIGNED'] = False

    taskIdx = 0
    subBaseVMS = baseVMS.copy()
    while not all(baseIoT.TASK_ASSIGNED):  # Loop until each existing TASK is assigned to a VM
        taskInf = baseIoT.iloc[taskIdx]
        
        # Get LOCAL Computation 
        Local_energy_consumption = (taskInf.N_INSTRUCTIONS / taskInf.IoT_MIpS) * taskInf.POWER * 1000
        Local_Power_Respect = True if (Local_energy_consumption/3600 < taskInf.POWER) else False
        if Local_Power_Respect:
            taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: ['LOCAL', 0, Local_energy_consumption]})], axis=1)
            taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : -1}, index=[0])], axis=1)
            baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
            taskIdx += 1 
            continue
        
        
        # Try assigning TASK on FOG
        best_FOG_VM, lowest_FOG_latency, FOG_TrmsnTime = compute_FOG_BestLatency(taskInf, subBaseVMS)
        FOG_energy_consumption = computeFOG_EC(FOG_TrmsnTime, taskInf, best_FOG_VM)
        FOG_PowerRespect = True if (FOG_energy_consumption/3600 < best_FOG_VM.POWER) else False
        FOG_ExecTimeRespect = True if (taskInf.N_INSTRUCTIONS / best_FOG_VM.VM_POWER_CAPACITY) < 10 else False 
        if FOG_PowerRespect and FOG_ExecTimeRespect:
            if best_FOG_VM.MEMORY_REMAINING - (taskInf.FILE_SIZE)/1000 > 0:  # MEMORY STORAGE is SUFFICIENT
                asctIndex = baseVMS[baseVMS.VM_ID == best_FOG_VM.VM_ID].index[0]
                baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
                baseVMS.loc[asctIndex, 'QUEUE_TASK'] = int(best_FOG_VM['QUEUE_TASK'] + 1)
                baseVMS.loc[asctIndex, 'QUEUE_TIME'] = float(best_FOG_VM['QUEUE_TIME'] + (taskInf.N_INSTRUCTIONS / best_FOG_VM.VM_POWER_CAPACITY))
                baseVMS.loc[asctIndex, 'MEMORY_REMAINING'] = best_FOG_VM['MEMORY_REMAINING'] - (taskInf.FILE_SIZE)/1000
                taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: [best_FOG_VM.VM_ID, lowest_FOG_latency, FOG_energy_consumption]})], axis=1)
                taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : 0}, index=[0])], axis=1)
                taskIdx += 1 
                subBaseVMS = baseVMS.copy()
                continue
            
        
        # Verify every FOG VMs have been tried
        if len(subBaseVMS[subBaseVMS.TYPE=='FOG'].VM_ID) != 1:
            subBaseVMS = subBaseVMS[subBaseVMS.VM_ID != best_FOG_VM.VM_ID].reset_index(drop=True)
            continue
            
        
        # Compute FOG - ENERGY CONSUMPTION
        best_CLOUD_VM, lowest_CLOUD_latency, CLOUD_TrmsnTime = compute_CLOUD_BestLatency(taskInf, baseVMS)
        CLOUD_energy_consumption = computeCLOUD_EC(FOG_TrmsnTime, CLOUD_TrmsnTime, taskInf, best_CLOUD_VM)

        asctIndex = baseVMS[baseVMS.VM_ID == best_CLOUD_VM.VM_ID].index[0]
        baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
        baseVMS.loc[asctIndex, 'QUEUE_TASK'] = int(best_CLOUD_VM['QUEUE_TASK'] + 1)
        baseVMS.loc[asctIndex, 'QUEUE_TIME'] = float(best_CLOUD_VM['QUEUE_TIME'] + (taskInf.N_INSTRUCTIONS / best_CLOUD_VM.VM_POWER_CAPACITY))
        taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: [best_CLOUD_VM.VM_ID, lowest_CLOUD_latency, CLOUD_energy_consumption]})], axis=1)
        taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : 1}, index=[0])], axis=1)
        taskIdx += 1 
        subBaseVMS = baseVMS.copy()
    
    
    # Modify Dummies
    taskVM_Dummy = taskVM_Dummy.loc[:, ~(taskVM_Dummy == -1).any()]
    taskVM = taskVM[list(taskVM_Dummy)]

    # Make Transpose
    taskVM_Transp = taskVM.transpose()
    taskVM_Transp = taskVM_Transp.reset_index()
    taskVM_Transp = taskVM_Transp.rename(columns={'index':'TASK', 0:'VM', 1:'F1', 2:'F2'})

    if saveInformation:
        # Save DATAFRAMES as CSV Files
        pathData = 'data/GENERATIONS'
        os.makedirs(pathData, exist_ok=True)
        lsFolders = os.listdir(pathData)
        folderGen = f'Generation_{int(max(list(map(lambda folderN : int(folderN.split("_")[1]), lsFolders)))) + 1}' if lsFolders else 'Generation_1'
        os.makedirs(os.path.join(pathData, folderGen), exist_ok=True)
        
        print(f'Data are stored within the following Folder → {folderGen}')
        baseVMS.to_csv(os.path.join(pathData, folderGen, f'BaseVMS_G{folderGen.split("_")[1]}.csv'))
        taskVM.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_G{folderGen.split("_")[1]}.csv'))
        taskVM_Dummy.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_DUMMY_G{folderGen.split("_")[1]}.csv'))
        taskVM_Transp.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_TRANSPOSE_G{folderGen.split("_")[1]}.csv'))

    return baseVMS, taskVM, taskVM_Dummy, taskVM_Transp, baseIoT



# -- Prepare Offloading - V3
def prepareOffloading_V3(n_IoTs = 20, FOG_nNodes = 5, CLOUD_nNodes = 3, IoT_rDist=((0,50), (0,50)), FOG_rDist = ((0,50), (0,50)), CLOUD_rDist = ((100, 500), (100, 500)), FOG_rVMS = (2, 6), CLOUD_rVMS = (2,4), saveInformation=False):
    taskVM, taskVM_Dummy = pd.DataFrame(), pd.DataFrame()
    
    baseVMS = setBaseVMS(FOG_nNodes, CLOUD_nNodes, FOG_rDist, CLOUD_rDist, FOG_rVMS, CLOUD_rVMS)
    baseVMS['TYPE'] = np.where(baseVMS['NODE_ID'].str.contains('F'), 'FOG', 'CLOUD')
    baseVMS['MEMORY_REMAINING'] = baseVMS['TOTAL_MEMORY']

    baseIoT, baseTasks = setEnvironment_IoT_n_TASKS(n_IoTs, IoT_rDist)
    baseIoT = baseIoT.merge(baseTasks, on='IoT_ID')
    baseIoT['TASK_ASSIGNED'] = False

    taskIdx = 0
    subBaseVMS = baseVMS.copy()
    while not all(baseIoT.TASK_ASSIGNED):  # Loop until each existing TASK is assigned to a VM
        taskInf = baseIoT.iloc[taskIdx]
        
        # Get LOCAL Computation 
        Local_energy_consumption = (taskInf.N_INSTRUCTIONS / taskInf.IoT_MIpS) * taskInf.POWER * 1000
        Local_Power_Respect = True if (Local_energy_consumption/3600 < taskInf.POWER) else False
        if Local_Power_Respect:
            taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: ['LOCAL', 0, Local_energy_consumption]})], axis=1)
            taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : -1}, index=[0])], axis=1)
            baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
            taskIdx += 1 
            continue
        
        
        # Try assigning TASK on FOG - DEPENDING ON VMS WITHOUT ANY ASSIGNED TASKS
        subBaseVMS_wthtQT = subBaseVMS[(subBaseVMS.TYPE=='FOG') & (subBaseVMS.QUEUE_TASK==0)]
        subTA1 = False
        while subTA1 is False:      
            if len(subBaseVMS_wthtQT) == 0:
                break   
            best_FOG_VM, lowest_FOG_latency, FOG_TrmsnTime = compute_FOG_BestLatency(taskInf, subBaseVMS_wthtQT.reset_index(drop=True))
            FOG_energy_consumption = computeFOG_EC(FOG_TrmsnTime, taskInf, best_FOG_VM)
            FOG_PowerRespect = True if (FOG_energy_consumption/3600 < best_FOG_VM.POWER) else False
            FOG_ExecTimeRespect = True if (lowest_FOG_latency) < 20 else False 
            if FOG_PowerRespect and FOG_ExecTimeRespect:
                if best_FOG_VM.MEMORY_REMAINING - (taskInf.FILE_SIZE)/1000 > 0:  # MEMORY STORAGE is SUFFICIENT                
                    subTA1 = True
                    break
            subBaseVMS_wthtQT = subBaseVMS_wthtQT[subBaseVMS_wthtQT.VM_ID != best_FOG_VM.VM_ID]

        if subTA1 is True:
            asctIndex = baseVMS[baseVMS.VM_ID == best_FOG_VM.VM_ID].index[0]
            baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
            baseVMS.loc[asctIndex, 'QUEUE_TASK'] = int(best_FOG_VM['QUEUE_TASK'] + 1)
            baseVMS.loc[asctIndex, 'QUEUE_TIME'] = best_FOG_VM['QUEUE_TIME'] + (taskInf.N_INSTRUCTIONS / best_FOG_VM.VM_POWER_CAPACITY)
            baseVMS.loc[baseVMS.NODE_ID==best_FOG_VM.NODE_ID, 'MEMORY_REMAINING'] = best_FOG_VM['MEMORY_REMAINING'] - (taskInf.FILE_SIZE)/1000
            taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: [best_FOG_VM.VM_ID, lowest_FOG_latency, FOG_energy_consumption]})], axis=1)
            taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : 0}, index=[0])], axis=1)
            taskIdx += 1 
            subBaseVMS = baseVMS.copy() 
            continue
                
        
        # Try assigning TASK on FOG - DEPENDING ON VMS WITHOUT ANY ASSIGNED TASKS
        subTA2 = False
        subBaseVMS_wthQT = subBaseVMS[(subBaseVMS.TYPE=='FOG') & (subBaseVMS.QUEUE_TASK!=0)]
        while subTA2 is False: 
            if len(subBaseVMS_wthQT) == 0:
                break
            best_FOG_VM, lowest_FOG_latency, FOG_TrmsnTime = compute_FOG_BestLatency(taskInf, subBaseVMS_wthQT.reset_index(drop=True))
            FOG_energy_consumption = computeFOG_EC(FOG_TrmsnTime, taskInf, best_FOG_VM)
            FOG_PowerRespect = True if (FOG_energy_consumption/3600 < best_FOG_VM.POWER) else False
            FOG_ExecTimeRespect = True if (lowest_FOG_latency) < 20 else False 
            if FOG_PowerRespect and FOG_ExecTimeRespect:
                if best_FOG_VM.MEMORY_REMAINING - (taskInf.FILE_SIZE)/1000 > 0:  # MEMORY STORAGE is SUFFICIENT
                    subTA2 = True
                    break
            subBaseVMS_wthQT = subBaseVMS_wthQT[subBaseVMS_wthQT.VM_ID != best_FOG_VM.VM_ID]
            
        if subTA2 is True:
            asctIndex = baseVMS[baseVMS.VM_ID == best_FOG_VM.VM_ID].index[0]
            baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
            baseVMS.loc[asctIndex, 'QUEUE_TASK'] = int(best_FOG_VM['QUEUE_TASK'] + 1)
            baseVMS.loc[asctIndex, 'QUEUE_TIME'] = best_FOG_VM['QUEUE_TIME'] + (taskInf.N_INSTRUCTIONS / best_FOG_VM.VM_POWER_CAPACITY)
            baseVMS.loc[baseVMS.NODE_ID==best_FOG_VM.NODE_ID, 'MEMORY_REMAINING'] = best_FOG_VM['MEMORY_REMAINING'] - (taskInf.FILE_SIZE)/1000
            taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: [best_FOG_VM.VM_ID, lowest_FOG_latency, FOG_energy_consumption]})], axis=1)
            taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : 0}, index=[0])], axis=1)
            taskIdx += 1 
            subBaseVMS = baseVMS.copy()
            continue
            
        # Compute FOG - ENERGY CONSUMPTION
        best_CLOUD_VM, lowest_CLOUD_latency, CLOUD_TrmsnTime = compute_CLOUD_BestLatency(taskInf, baseVMS)
        CLOUD_energy_consumption = computeCLOUD_EC(FOG_TrmsnTime, CLOUD_TrmsnTime, taskInf, best_CLOUD_VM)

        asctIndex = baseVMS[baseVMS.VM_ID == best_CLOUD_VM.VM_ID].index[0]
        baseIoT.loc[taskIdx, 'TASK_ASSIGNED'] = True
        baseVMS.loc[asctIndex, 'QUEUE_TASK'] = int(best_CLOUD_VM['QUEUE_TASK'] + 1)
        baseVMS.loc[asctIndex, 'QUEUE_TIME'] = best_CLOUD_VM['QUEUE_TIME'] + (taskInf.N_INSTRUCTIONS / best_CLOUD_VM.VM_POWER_CAPACITY)
        taskVM = pd.concat([taskVM, pd.DataFrame({taskInf.TASK_ID: [best_CLOUD_VM.VM_ID, lowest_CLOUD_latency, CLOUD_energy_consumption]})], axis=1)
        taskVM_Dummy = pd.concat([taskVM_Dummy, pd.DataFrame({taskInf.TASK_ID : 1}, index=[0])], axis=1)
        taskIdx += 1 
        subBaseVMS = baseVMS.copy()
        
    
    # Make Transpose
    taskVM_Transp = taskVM.transpose()
    taskVM_Transp = taskVM_Transp.reset_index()
    taskVM_Transp = taskVM_Transp.rename(columns={'index':'TASK', 0:'VM', 1:'F1', 2:'F2'})



    if saveInformation:
        # Save DATAFRAMES as CSV Files
        pathData = 'data/GENERATIONS'
        os.makedirs(pathData, exist_ok=True)
        lsFolders = os.listdir(pathData)
        folderGen = f'Generation_{int(max(list(map(lambda folderN : int(folderN.split("_")[1]), lsFolders)))) + 1}' if lsFolders else 'Generation_1'
        os.makedirs(os.path.join(pathData, folderGen), exist_ok=True)
        
        print(f'Data are stored within the following Folder → {folderGen}')
        baseVMS.to_csv(os.path.join(pathData, folderGen, f'BaseVMS_G{folderGen.split("_")[1]}.csv'))
        taskVM.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_G{folderGen.split("_")[1]}.csv'))
        taskVM_Dummy.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_DUMMY_G{folderGen.split("_")[1]}.csv'))
        taskVM_Transp.to_csv(os.path.join(pathData, folderGen, f'TASKS_VMS_TRANSPOSE_G{folderGen.split("_")[1]}.csv'))

    return baseVMS, taskVM, taskVM_Dummy, taskVM_Transp



# - CORE -  
if __name__ == '__main__':
    prepareOffloading_V2()