import numpy as np
import pandas as pd
# Initialize a 3x10 matrix with zeros
serialbuff = np.zeros((3, 10))
print(pd.DataFrame(serialbuff))
# List of new rows to append
new_rows = [[0, 1, 2], [2, 3, 2]]
print(pd.DataFrame(serialbuff[:,1:]))# Iterate over each new row
# for i in new_rows:
#     # Append the new row to the bottom of the matrix
#     serialbuff = np.append(serialbuff[:,1:], i)
    
#     # Remove the first row of the matrix

#     # Print the updated matrix
#     print(pd.DataFrame(serialbuff))