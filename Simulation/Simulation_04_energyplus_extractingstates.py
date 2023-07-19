import csv
import pandas as pd

# Make an empty dataframe to save the states in.
df_state = pd.DataFrame()

# for i in range (1):
#     for j in range (8):
#
#         # read the CSV file into a DataFrame
#         df_file = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                              f"02_Working/Export/20230419_113656/edit_output_{i}_{j * 45}.csv")
#
#         df_state[f'state_shade_{i}_north_{j*45}'] = df_file['state_index']
#
# for i in range (1):
#     for j in range (8):
#
#         # read the CSV file into a DataFrame
#         df_file = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                              f"02_Working/Export/20230419_113656/edit_output_{i+6}_{j * 45}.csv")
#
#         df_state[f'state_shade_{i+6}_north_{j*45}'] = df_file['state_index']
#
# print(df_state)
#
# # Replace non-finite values with 0
# df_state.fillna(0, inplace=True)
#
# # Convert the values in the DataFrame to integers
# df_state = df_state.astype(int)
#
# df_state_transposed = df_state.transpose()
#
# print(df_state_transposed)
#
# df_state_transposed.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                     f"02_Working/Export/20230419_113656/sim_states.xlsx"), header=False, index=False)

