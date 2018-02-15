import numpy as np

def Binning(X,arguments=None):
            bins=[int(b) for b in arguments.split(",")]
            binValues=[]
            i=0
            for x in X:
                i=0
                while i<len(bins):                    
                    if (x<=bins[i]) or (i==len(bins)-1 and x>bins[i]):
                        binValues.append(i)
                        break                        
                    elif i<len(bins)-1 and x>bins[i] and x<=bins[i+1]:
                        binValues.append(i+1)
                        break                        
                    i=i+1
            return np.array(binValues)
        
