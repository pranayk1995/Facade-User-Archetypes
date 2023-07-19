import pandas as pd

# Make an empty dataframe to save the states in.
df_heating = pd.DataFrame()
df_cooling = pd.DataFrame()
df_temp = pd.DataFrame()
df_light = pd.DataFrame()
df_udi = pd.DataFrame()
df_glare = pd.DataFrame()
df_heating_noshadez = pd.DataFrame()
df_cooling_noshadez = pd.DataFrame()
df_temp_noshadez = pd.DataFrame()
df_light_noshadez = pd.DataFrame()
df_energy_noshadez = pd.DataFrame()

# for i in range(6):
#     for j in range(8):
#
#         # read the CSV file into a DataFrame
#         df_file = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                               f"02_Working/Export/20230419_113656/edit_output_{i}_{j * 45}.csv")
#
#         df_heating[f'Heating_{i}_{j*45}'] = df_file['heating_energy']
#
#         df_cooling[f'Cooling_{i}_{j*45}'] = df_file['cooling_energy']
#
#         df_temp[f'Operative_temp_{i}_{j*45}'] = df_file['operative_temperature']
#
#         df_light[f'Lighting_{i}_{j*45}'] = df_file['lighting_energy']
#
# for i in range(6):
#     for j in range(8):
#
#         # read the CSV file into a DataFrame
#         df_file = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                               f"02_Working/Export/20230419_113656/edit_output_{i+6}_{j * 45}.csv")
#
#         df_heating[f'Heating_{i+6}_{j*45}'] = df_file['heating_energy']
#
#         df_cooling[f'Cooling_{i+6}_{j*45}'] = df_file['cooling_energy']
#
#         df_temp[f'Operative_temp_{i+6}_{j*45}'] = df_file['operative_temperature']
#
#         df_light[f'Lighting_{i+6}_{j*45}'] = df_file['lighting_energy']
#
# # Convert the values in the DataFrame to integers
# df_heating = df_heating.astype(int)
# df_cooling = df_cooling.astype(int)
# df_temp = df_temp.astype(float)
# df_light = df_light.astype(float)
#
# df_heating.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                     f"02_Working/Export/20230419_113656/00_heating.xlsx"), header=True, index=False)
#
# df_cooling.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                     f"02_Working/Export/20230419_113656/00_cooling.xlsx"), header=True, index=False)
#
# df_temp.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                   f"02_Working/Export/20230419_113656/00_temp.xlsx"), header=True, index=False)
#
# df_light.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                   f"02_Working/Export/20230419_113656/00_light.xlsx"), header=True, index=False)

for j in range(8):
    # read the CSV file into a DataFrame
    df_file = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                          f"02_Working/Export/20230516_173425/edit_output_noshade_edited_{j * 45}.csv")

    df_heating_noshadez[f'Heating_{j * 45}'] = df_file['heating_energy']

    df_cooling_noshadez[f'Cooling_{j * 45}'] = df_file['cooling_energy']

    df_temp_noshadez[f'Operative_temp_{j * 45}'] = df_file['operative_temperature']

    df_light_noshadez[f'Lighting_{j * 45}'] = df_file['lighting_energy']

    df_energy_noshadez[f'energy_{j * 45}'] = df_file['total_energy']

# Convert the values in the DataFrame to integers
df_heating_noshadez = df_heating_noshadez.astype(int)
df_cooling_noshadez = df_cooling_noshadez.astype(int)
df_temp_noshadez = df_temp_noshadez.astype(float)
df_light_noshadez = df_light_noshadez.astype(float)
df_energy_noshadez = df_energy_noshadez.astype(float)

df_heating_noshadez.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                    f"02_Working/Export/20230516_173425/00_heating.xlsx"), header=True, index=False)

df_cooling_noshadez.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                    f"02_Working/Export/20230516_173425/00_cooling.xlsx"), header=True, index=False)

df_temp_noshadez.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                  f"02_Working/Export/20230516_173425/00_temp.xlsx"), header=True, index=False)

df_light_noshadez.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                  f"02_Working/Export/20230516_173425/00_light.xlsx"), header=True, index=False)

df_energy_noshadez.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                  f"02_Working/Export/20230516_173425/00_energy.xlsx"), header=True, index=False)