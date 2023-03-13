####################################################
#  Optimization Metaheuristic - Algorithm NSGA-II  #
####################################################

# - LIBRARIES - 

# -- General --
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.set_option('mode.chained_assignment', None)

# -- Script based Packages --
from setPopulations import *


# - FUNCTIONS - 

# -- --
def fast_non_dominated_sort(df_coords):
    rank = 1
    df_coords['RANK'] = None
    while None in set(df_coords['RANK']):
        df_coords_slct = df_coords[df_coords['RANK'].isnull()]
        dict_ = {}
        for i, row in df_coords_slct.iterrows():
            lsPareto = [0]
            for i_bis, row_bis in df_coords_slct.iterrows():
                if i == i_bis:
                    continue
                if row.F1 < row_bis.F1 or row.F2 < row_bis.F2:
                    lsPareto.append(1)
            dict_[row['Population']] = sum(lsPareto)
        lsVal = list(filter(lambda key_ : dict_[key_] == max(dict_.values()), dict_.keys()))
        for point_ in lsVal:
            df_coords['RANK'][df_coords[df_coords['Population']==point_].index[0]] = rank
        rank += 1
    return df_coords

# -- --
def crowding_distance(population_df_ranked):
    population_df_ranked['CROWDING_DISTANCE'] = 0
    unique_rank = sorted(population_df_ranked['RANK'].unique().tolist())

    concat_df = pd.DataFrame()
    for rank in unique_rank:
        sub_pop = population_df_ranked[population_df_ranked.RANK == rank]
        for obj in ['F1', 'F2']:
            sub_pop = sub_pop.sort_values(by=obj, ascending=True).reset_index(drop=True)
            sub_pop_len = len(sub_pop) - 1   
            
            sub_pop.loc[0,'CROWDING_DISTANCE'] = float("inf")
            sub_pop.loc[sub_pop_len,'CROWDING_DISTANCE'] = float("inf")
            for i in range(1, sub_pop_len):
                if sub_pop.loc[0,obj] == sub_pop.loc[sub_pop_len,obj]:
                    sub_pop['CROWDING_DISTANCE'] = 0
                    break
                sub_pop.loc[i,'CROWDING_DISTANCE'] += (
                        sub_pop.loc[i+1,obj] - sub_pop.loc[i-1,obj]
                    ) / (sub_pop.loc[sub_pop_len,obj] - sub_pop.loc[0,obj])
        concat_df = pd.concat([concat_df, sub_pop]).reset_index(drop=True)
    return concat_df


# -- --
def tournament_selection(population, TOURNAMENT_SIZE):
    tournament = population.sample(TOURNAMENT_SIZE)
    tournament_crowding = crowding_distance(tournament)
    best_candidate = tournament_crowding.iloc[tournament_crowding.sample().index[0]]
    for i in range(TOURNAMENT_SIZE):
        parent = tournament_crowding.iloc[i]
        if parent.RANK < best_candidate.RANK:
            best_candidate = parent
        elif parent.RANK == best_candidate.RANK:
            if parent.CROWDING_DISTANCE > best_candidate.CROWDING_DISTANCE:
                best_candidate = parent
    return best_candidate.to_frame().transpose()


# -- --
def croisement(parent_1, parent_2, sol, sol_dum):
    
    parent_1_dum = [item for item in sol_dum if int(item[0]) == int(parent_1["ID"])][0][1]
    parent_2_dum = [item for item in sol_dum if int(item[0]) == int(parent_2["ID"])][0][1]

    parent_1_details = [item for item in sol if int(item[0]) == int(parent_1["ID"])][0][1]
    parent_2_details = [item for item in sol if int(item[0]) == int(parent_2["ID"])][0][1]

    len_chromosome = len(parent_1_dum.columns)
    cross_point = random.randint(0, len_chromosome - 1)

    parent_1_a_dum = parent_1_dum[parent_1_dum.columns[:cross_point]]
    parent_1_b_dum = parent_1_dum[parent_1_dum.columns[cross_point:]]
    parent_2_a_dum = parent_2_dum[parent_2_dum.columns[:cross_point]]
    parent_2_b_dum = parent_2_dum[parent_2_dum.columns[cross_point:]]

    child_1_dum = pd.concat([parent_1_a_dum,parent_2_b_dum], axis=1)
    child_2_dum = pd.concat([parent_2_a_dum,parent_1_b_dum], axis=1)

    parent_1_a_details = parent_1_details[parent_1_details.columns[:cross_point]]
    parent_1_b_details = parent_1_details[parent_1_details.columns[cross_point:]]
    parent_2_a_details = parent_2_details[parent_2_details.columns[:cross_point]]
    parent_2_b_details = parent_2_details[parent_2_details.columns[cross_point:]]

    child_1_details = pd.concat([parent_1_a_details,parent_2_b_details], axis=1)
    child_2_details = pd.concat([parent_2_a_details,parent_1_b_details], axis=1)

    return child_1_dum, child_2_dum, child_1_details, child_2_details


# -- --
def mutation(baseVMS, baseIoT, detailsChild, populationID):    
    baseIoT = random.choice(baseIoT)
    
    randTask = random.choice(list(detailsChild))
    taskInfo = baseIoT[baseIoT.TASK_ID == randTask].reset_index(drop=True).iloc[0]
    initVM = 0 if ('F' in detailsChild[randTask][0]) else 1  # 0 if FOG / 1 if CLOUD
    updateVM = abs(int(initVM) - 1)


    if updateVM == 0:
        baseVMS = random.choice(baseVMS)
        bVMS = baseVMS[baseVMS.NODE_ID.str.contains('F')].sample().reset_index(drop=True).iloc[0]
        QTime = 0 if bVMS.QUEUE_TIME is None else bVMS.QUEUE_TIME  # Queue Time
        PgtTime = computePropagationTime((taskInfo.X, taskInfo.Y), (bVMS.X, bVMS.Y)) # Propagation Time
        TrmsnTime = computeTransmissionTime(taskInfo.FILE_SIZE, bVMS.VM_BANDWITH) # Transmission Time
        LATENCY = QTime + PgtTime + TrmsnTime
        EC = computeFOG_EC(TrmsnTime, taskInfo, bVMS)
        
    else:
        for contentBaseVMS in baseVMS:
            if str(detailsChild[randTask].iloc[0]) in contentBaseVMS.VM_ID.unique():
                baseVMS = contentBaseVMS
                break
        bVMS_F = baseVMS[baseVMS.VM_ID == detailsChild[randTask].iloc[0]].reset_index(drop=True).iloc[0]
        F_TrmsnTime = computeTransmissionTime(taskInfo.FILE_SIZE, bVMS_F.VM_BANDWITH) # Transmission Time
        
        bVMS = baseVMS[baseVMS.NODE_ID.str.contains('C')].sample().reset_index(drop=True).iloc[0]
        QTime = 0 if bVMS.QUEUE_TIME is None else bVMS.QUEUE_TIME  # Queue Time
        PgtTime = computePropagationTime((taskInfo.X, taskInfo.Y), (bVMS.X, bVMS.Y)) # Propagation Time
        TrmsnTime = computeTransmissionTime(taskInfo.FILE_SIZE, bVMS.VM_BANDWITH) # Transmission Time
        LATENCY = QTime + PgtTime + TrmsnTime
        EC = computeCLOUD_EC(F_TrmsnTime, TrmsnTime, taskInfo, bVMS)   

    detailsChild[randTask].loc[0] = bVMS.VM_ID
    detailsChild[randTask].loc[1] = LATENCY
    detailsChild[randTask].loc[2] = EC

    Child = getTotal_fromParentsSolutions([detailsChild]).transpose().reset_index(names='Population').rename(columns={0: "ID",1:'F1', 2:'F2'})
    Child['Population'] = f'Population_{populationID}'
    Child["ID"] = populationID
    
    return Child, detailsChild



def NSGA_II(soluce, soluce_dummy, soluce_total, soluce_VMs, soluce_IoTs, TOURNAMENT_SIZE=2, CROSSOVER_PROBABILITY=0.9, MUTATION_PROBABILITY=0.1, N_GENERATIONS=30):

    POPULATION = round(np.mean([len(soluce), len(soluce_dummy)]))
    sub_soluce = list(map(lambda sol : sol, enumerate(soluce)))
    sub_soluce_dum = list(map(lambda sol_d : sol_d, enumerate(soluce_dummy)))

    population_df_T = soluce_total.transpose()
    population_df_T = population_df_T.reset_index(names='Population')
    population_df_T = population_df_T.rename(columns={0: "ID",1:'F1', 2:'F2'})
    population_df_T.F1 = population_df_T.F1.astype('float')
    population_df_T.F2 = population_df_T.F2.astype('float')

    population = population_df_T.copy()
    population_df_ranked = fast_non_dominated_sort(population).reset_index(drop=True)

    for i in range(N_GENERATIONS):
        print(f'Current Generation -> {i+1}')
        
        counter = 0
        child_sol = []
        child_sol_dum = []
        child_sol_total = pd.DataFrame()
        while len(child_sol_total) < POPULATION:
            parent_1 = tournament_selection(population_df_ranked, TOURNAMENT_SIZE)
            parent_2 = tournament_selection(population_df_ranked, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_PROBABILITY:
                child_1_dum, child_2_dum, child_1_details, child_2_details = croisement(parent_1, parent_2, sub_soluce, sub_soluce_dum)
                child_1 = getTotal_fromParentsSolutions([child_1_details]).transpose().reset_index(names='Population').rename(columns={0: "ID",1:'F1', 2:'F2'})
                counter += 1
                child_1['Population'] = 'Population_' + str((i + 1) * POPULATION + counter)
                child_1["ID"] = (i + 1) * POPULATION + counter

                child_sol.append((child_1["ID"], child_1_details))
                child_sol_dum.append((child_1["ID"],child_1_dum))

                child_2 = getTotal_fromParentsSolutions([child_2_details]).transpose().reset_index(names='Population').rename(columns={0: "ID",1:'F1', 2:'F2'})
                counter += 1
                child_2['Population'] = 'Population_' + str((i + 1) * POPULATION + counter)
                child_2["ID"] = (i + 1) * POPULATION + counter
                
                child_sol.append((child_2["ID"],child_2_details))        
                child_sol_dum.append((child_2["ID"],child_2_dum))
            else:
                child_1 = parent_1
                child_2 = parent_2

                child_1_dum = [item for item in sub_soluce_dum if int(item[0]) == int(parent_1["ID"])][0][1]
                child_2_dum = [item for item in sub_soluce_dum if int(item[0]) == int(parent_2["ID"])][0][1]

                child_1_details = [item for item in sub_soluce if int(item[0]) == int(parent_1["ID"])][0][1]
                child_2_details = [item for item in sub_soluce if int(item[0]) == int(parent_2["ID"])][0][1]
                
                counter += 1
                child_1['Population'] = 'Population_' + str((i + 1) * POPULATION + counter)
                child_1["ID"] = (i + 1) * POPULATION + counter

                child_sol.append((child_1["ID"], child_1_details))
                child_sol_dum.append((child_1["ID"],child_1_dum))

                counter += 1
                child_2['Population'] = 'Population_' + str((i + 1) * POPULATION + counter)
                child_2["ID"] = (i + 1) * POPULATION + counter

                child_sol.append((child_2["ID"],child_2_details))        
                child_sol_dum.append((child_2["ID"],child_2_dum))

            if random.random() < MUTATION_PROBABILITY:      
                try:      
                    child_1, child_1_details = mutation(soluce_VMs, soluce_IoTs, child_1_details, int(child_1.Id.values[0]))
                    child_2, child_2_details = mutation(soluce_VMs, soluce_IoTs, child_2_details, int(child_2.Id.values[0]))
                except:
                    pass
        
            child_sol_total = pd.concat([child_sol_total,child_1, child_2]).reset_index(drop=True)
        parent_child_sol_total = pd.concat([population_df_ranked, child_sol_total]).reset_index(drop=True)

        parent_child_sol_total_copy = parent_child_sol_total.copy()
        parent_child_sorted = fast_non_dominated_sort(parent_child_sol_total_copy).reset_index(drop=True)
        parent_child_sorted_distance = crowding_distance(parent_child_sorted)
        new_population = parent_child_sorted_distance.sort_values(by=['RANK','CROWDING_DISTANCE'], ascending=[True, False])[:POPULATION]
        new_population["ID"] = new_population["ID"].astype(int)
        
        new_soluce = []
        sol_id_list = [int(item[0]) for item in sub_soluce]
        child_id_list = [int(item[0]) for item in child_sol]
        for id in new_population["ID"]:
            if id in sol_id_list:
                sol_item = [item for item in sub_soluce if int(item[0]) == id][0]
                new_soluce.append(sol_item)
            elif id in child_id_list:
                child_item = [item for item in child_sol if int(item[0]) == id][0]
                new_soluce.append(child_item)
        
        sub_soluce = new_soluce.copy()

        new_sol_dum = []
        sol_id_list_dum = [int(item[0]) for item in sub_soluce_dum]
        child_id_list_dum = [int(item[0]) for item in child_sol_dum]
        for id in new_population["ID"]:
            id = int(id)     
            if id in sol_id_list_dum:
                sol_item = [item for item in sub_soluce_dum if int(item[0]) == id][0]
                new_sol_dum.append(sol_item)
            elif id in child_id_list_dum:
                child_item = [item for item in child_sol_dum if int(item[0]) == id][0]
                new_sol_dum.append(child_item)
        sub_soluce_dum = new_sol_dum.copy()
        population_df_ranked = new_population.copy()

    return sub_soluce, sub_soluce_dum, population_df_ranked