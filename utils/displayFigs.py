##################################################
#  Optimization Metaheuristic - Display Figures  #
##################################################

# - LIBRARIES - 

# -- General --
import matplotlib.pyplot as plt

# -- Script based Packages --
from NSGA_II import fast_non_dominated_sort


# - FUNCTIONS - 

# -- display solutions --
def displaySolutions(dfSolutions):
    _, ax = plt.subplots(figsize=(10, 10))
    for _, row in dfSolutions.iterrows():
        ax.annotate(row['Population'], (row['F1']+0.02, row['F2']), fontsize=10)
        if row.RANK == 1:
            ax.scatter(row['F1'], row['F2'], color='red')
        else:
            ax.scatter(row['F1'], row['F2'], color='blue')
    plt.xlabel('LATENCY')
    plt.ylabel('ENERGY CONSUMPTION')
    plt.show()


# -- display content from the optimal solutions
def displayOptimalContent(dfSolutions, lsSolutions):
    db_optPop = [item for item in lsSolutions if int(item[0]) == dfSolutions.iloc[0].ID][0][1]
    db_optPop_T = db_optPop.transpose()
    db_optPop_T = db_optPop_T.reset_index(names='Population')
    db_optPop_T = db_optPop_T.rename(columns={0: 'VM',1:'F1', 2:'F2'})

    _, ax = plt.subplots(figsize=(15, 10))
    for _, row in db_optPop_T.iterrows():
        ax.annotate(row['VM'], (row['F1']+0.02, row['F2']), fontsize=10)
        if 'C' in row.VM:
            ax.scatter(row['F1'], row['F2'], color='red')
        else:
            ax.scatter(row['F1'], row['F2'], color='blue')
    plt.xlabel('LATENCY')
    plt.ylabel('ENERGY CONSUMPTION')
    plt.show()