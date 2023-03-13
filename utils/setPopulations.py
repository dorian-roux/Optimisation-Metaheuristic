#####################################################
#  Optimization Metaheuristic - Prepare Population  #
#####################################################

# - LIBRARIES - 

# -- General --
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# -- Script based Packages --
from tasksOffloading import *


# - FUNCTIONS - 

# -- --
def getTotal_fromParentsSolutions(solutions):
    dfSols = pd.DataFrame()
    for i, sol in enumerate(solutions):
        sumLATENCY = sum(sol.iloc[1].reset_index(drop=True).tolist())
        sumEC = sum(sol.iloc[2].reset_index(drop=True).tolist())
        dfSols = pd.concat([dfSols, pd.DataFrame({f'Population_{i}': [i, sumLATENCY, sumEC]})], axis=1)    
    return dfSols


# -- --
def makeTranpose(baseData, nameIndex, lsColumns):
            if isinstance(nameIndex, str):
                baseData = baseData.transpose().reset_index(names=nameIndex)
            if not isinstance(lsColumns, list) or (len(lsColumns) != len(baseData.columns)):
                return baseData
            baseData.columns = lsColumns
            return baseData
       
# -- --
def constructPopulation(popSize=10, n_IoTs=20, FOG_nNodes=5, CLOUD_nNodes=3, IoT_rDist=((0,100), (0,100)), FOG_rDist=((0,100), (0, 100)), CLOUD_rDist=((100,500), (100,500)), FOG_rVMS=(2,6), CLOUD_rVMS=(2,3)):
    ls_Solutions, ls_Solutions_T, ls_Solutions_Dummy, ls_Solutions_VMS, ls_Solutions_IoT = [], [], [], [], [] 
    for _ in range(popSize):
        baseVMS, taskVM, taskVM_Dummy, taskVM_Transp, baseIoT = prepareOffloading_V2(n_IoTs, FOG_nNodes, CLOUD_nNodes, IoT_rDist, FOG_rDist, CLOUD_rDist, FOG_rVMS, CLOUD_rVMS, False)   
        ls_Solutions.append(taskVM)
        ls_Solutions_T.append(taskVM_Transp)
        ls_Solutions_Dummy.append(taskVM_Dummy)
        ls_Solutions_VMS.append(baseVMS)
        ls_Solutions_IoT.append(baseIoT)
    return ls_Solutions, ls_Solutions_T, ls_Solutions_Dummy,  getTotal_fromParentsSolutions(ls_Solutions), ls_Solutions_VMS, ls_Solutions_IoT



# - CORE -
if __name__ == '__main__':
    constructPopulation()