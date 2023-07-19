import pandas as pd

# for i in range(6):
#     for j in range(8):
#
#         # read the CSV file into a DataFrame
#         df = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                          f"02_Working/Export/20230419_113656/output_{i}_{j*45}.csv")
#
#         # create a new column that is the product of two existing columns
#         df["state_index"] = df["APERTURE_0BEF98DD:Surface Shading Device Is On Time Fraction [](Hourly)"]
#
#         # remove the column(s) you don't want
#         df = df.drop(columns=['Date/Time',
#                               'ROOM_453_5DC66EC0:Zone Total Internal Total Heating Energy [J](Hourly)',
#                               'APERTURE_0BEF98DD:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_0BEF98DD:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_3EC4A215:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_3EC4A215:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_75F2FD86:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_75F2FD86:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_E24BABA1:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_E24BABA1:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_F3066094:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_F3066094:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'ROOM_453_5DC66EC0:Zone Adaptive Comfort Operative Temperature Set Point [C](Hourly)',
#                               'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)',
#                               'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)',
#                               'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)'])
#
#         df = df.rename(columns={'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)': 'drybulb_temperature',
#                                 'Environment:Site Outdoor Air Relative Humidity [%](Hourly)': 'relative_humidity',
#                                 'WATTS_ROOM1:Lights Electricity Energy [J](Hourly)': 'lighting_energy',
#                                 'ROOM_453_5DC66EC0:Zone Mean Radiant Temperature [C](Hourly)': 'mrt',
#                                 'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)': 'thermostat',
#                                 'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](Hourly)': 'heating_energy',
#                                 'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](Hourly) ': 'cooling_energy',
#                                 'ROOM_453_5DC66EC0:Zone Operative Temperature [C](Hourly)': 'operative_temperature'})
#
#         # save the modified DataFrame to a new CSV file
#         df.to_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                   f"02_Working/Export/20230419_113656/edit_output_{i}_{j*45}.csv", index=False)
#
# for i in range(6):
#     for j in range(8):
#
#         # read the CSV file into a DataFrame
#         df = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                          f"02_Working/Export/20230419_113656/output_{i+6}_{j*45}.csv")
#
#
#         # Define the function to map values to the desired index
#         def map_angle_to_index(angle):
#             if angle > 82.5:
#                 return 1
#             elif angle >= 67.5 and angle < 82.5:
#                 return 2
#             elif angle >= 52.5 and angle < 67.5:
#                 return 3
#             elif angle >= 37.5 and angle < 52.5:
#                 return 4
#             elif angle >= 22.5 and angle < 37.5:
#                 return 5
#             elif angle >= 7.5 and angle < 22.5:
#                 return 6
#             elif angle > 0 and angle < 7.5:
#                 return 7
#             else:
#                 return 0
#
#         # # define the bins and categories
#         # bins = [-1, 0, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 90]
#         # categories = [0, 1, 2, 3, 4, 5, 6, 7]
#
#         # # use pd.cut to create the new column
#         # df['state_index'] = pd.cut(df['APERTURE_0BEF98DD:Surface Window Blind Slat Angle [deg](Hourly)'],
#         #                            bins=bins, labels=categories, include_lowest=True)
#
#         # Apply the function to the original column and create the new column
#         df['state_index'] = df['APERTURE_0BEF98DD:Surface Window Blind Slat Angle [deg](Hourly)'].apply(map_angle_to_index)
#
#         # remove the column(s) you don't want
#         df = df.drop(columns=['Date/Time',
#                               'ROOM_453_5DC66EC0:Zone Total Internal Total Heating Energy [J](Hourly)',
#                               'APERTURE_0BEF98DD:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_0BEF98DD:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_3EC4A215:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_3EC4A215:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_75F2FD86:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_75F2FD86:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_E24BABA1:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_E24BABA1:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'APERTURE_F3066094:Surface Shading Device Is On Time Fraction [](Hourly)',
#                               'APERTURE_F3066094:Surface Window Blind Slat Angle [deg](Hourly)',
#                               'ROOM_453_5DC66EC0:Zone Adaptive Comfort Operative Temperature Set Point [C](Hourly)',
#                               'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)',
#                               'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)',
#                               'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)'])
#
#         df = df.rename(columns={'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)': 'drybulb_temperature',
#                                 'Environment:Site Outdoor Air Relative Humidity [%](Hourly)': 'relative_humidity',
#                                 'WATTS_ROOM1:Lights Electricity Energy [J](Hourly)': 'lighting_energy',
#                                 'ROOM_453_5DC66EC0:Zone Mean Radiant Temperature [C](Hourly)': 'mrt',
#                                 'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)': 'thermostat',
#                                 'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](Hourly)': 'heating_energy',
#                                 'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](Hourly) ': 'cooling_energy',
#                                 'ROOM_453_5DC66EC0:Zone Operative Temperature [C](Hourly)': 'operative_temperature'})
#
#         # save the modified DataFrame to a new CSV file
#         df.to_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
#                   f"02_Working/Export/20230419_113656/edit_output_{i+6}_{j*45}.csv", index=False)



for j in range(8):

    # read the CSV file into a DataFrame
    df_noshades = pd.read_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                              f"02_Working/Export/20230516_173425/output_noshade_{j*45}.csv")

    df_noshades = df_noshades.drop(columns=['Date/Time',
                              'ROOM_453_5DC66EC0:Zone Total Internal Total Heating Energy [J](Hourly)',
                              'APERTURE_0BEF98DD:Surface Shading Device Is On Time Fraction [](Hourly)',
                              'APERTURE_0BEF98DD:Surface Window Blind Slat Angle [deg](Hourly)',
                              'APERTURE_3EC4A215:Surface Shading Device Is On Time Fraction [](Hourly)',
                              'APERTURE_3EC4A215:Surface Window Blind Slat Angle [deg](Hourly)',
                              'APERTURE_75F2FD86:Surface Shading Device Is On Time Fraction [](Hourly)',
                              'APERTURE_75F2FD86:Surface Window Blind Slat Angle [deg](Hourly)',
                              'APERTURE_E24BABA1:Surface Shading Device Is On Time Fraction [](Hourly)',
                              'APERTURE_E24BABA1:Surface Window Blind Slat Angle [deg](Hourly)',
                              'APERTURE_F3066094:Surface Shading Device Is On Time Fraction [](Hourly)',
                              'APERTURE_F3066094:Surface Window Blind Slat Angle [deg](Hourly)',
                              'ROOM_453_5DC66EC0:Zone Adaptive Comfort Operative Temperature Set Point [C](Hourly)',
                              'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy [J](Hourly)',
                              'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Cooling Energy [J](Hourly)',
                              'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)'])

    df_noshades = df_noshades.rename(columns={'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)': 'drybulb_temperature',
                            'Environment:Site Outdoor Air Relative Humidity [%](Hourly)': 'relative_humidity',
                            'WATTS_ROOM1:Lights Electricity Energy [J](Hourly)': 'lighting_energy',
                            'ROOM_453_5DC66EC0:Zone Mean Radiant Temperature [C](Hourly)': 'mrt',
                            'ROOM_453_5DC66EC0:Zone Thermostat Air Temperature [C](Hourly)': 'thermostat',
                            'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Heating Energy [J](Hourly)': 'heating_energy',
                            'ROOM_453_5DC66EC0 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Energy [J](Hourly) ': 'cooling_energy',
                            'ROOM_453_5DC66EC0:Zone Operative Temperature [C](Hourly)': 'operative_temperature'})

    df_noshades['total_energy'] = df_noshades['heating_energy'] + df_noshades['cooling_energy'] + df_noshades['lighting_energy']

    # save the modified DataFrame to a new CSV file
    df_noshades.to_csv(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                       f"02_Working/Export/20230516_173425/edit_output_noshade_edited_{j * 45}.csv", index=False)