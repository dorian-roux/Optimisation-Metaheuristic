##################################################
#  Optimization Metaheuristic - Display Results  #
##################################################

# - LIBRARIES - 

# -- General --
import os
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


# - FUNCTIONS - 

# get Rank
def getParetoRank(taskVM_TRANSPOSE):
    rank_ = 1
    taskVM_TRANSPOSE['RANK'] = None

    while None in set(taskVM_TRANSPOSE['RANK']):
        df_coords_slct = taskVM_TRANSPOSE[taskVM_TRANSPOSE['RANK'].isnull()]
        dict_ = {}
        for i, row in df_coords_slct.iterrows():
            lsPareto = [0]
            for i_bis, row_bis in df_coords_slct.iterrows():
                if i == i_bis:
                    continue
                
                if row.F1 < row_bis.F1 or row.F2 < row_bis.F2:
                    lsPareto.append(1)

            dict_[row['TASK']] = sum(lsPareto)

        lsVal = list(filter(lambda key_ : dict_[key_] == max(dict_.values()), dict_.keys()))
        for point_ in lsVal:
            taskVM_TRANSPOSE['RANK'][taskVM_TRANSPOSE[taskVM_TRANSPOSE['TASK']==point_].index[0]] = rank_
        rank_ += 1
    return taskVM_TRANSPOSE