import datetime
from eppy.modeleditor import IDF
import os

# Set the path to the EnergyPlus executable
eplus_path = "C:/EnergyPlusV22-2-0/EnergyPlus.exe"

# Set the path to the IDD file
iddfile = "C:/EnergyPlusV22-2-0/Energy+.idd"

# Load the IDF file
idf_file = "C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
           "02_Working/EnergyPlus/20230419_T8_ep_file/Forloop.idf"
epw_file = "C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
           "02_Working/epw/NLD_Amsterdam.062400_IWEC/NLD_Amsterdam.062400_IWEC.epw"

IDF.setiddname(iddfile)
idf1 = IDF(idf_file, epw_file)

building = idf1.idfobjects['BUILDING'][0]
print(building.Name)

# Get a single WINDOWCONSTRUCTION object
window_const = idf1.idfobjects["CONSTRUCTION"][22]
print(f"{window_const}")

# Get a single WINDOWMATERIAL:SHADE AND WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM
window_mat_g = idf1.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][0]
print(f"Name: {window_mat_g.Name}")

# Update the Outside_Layer and Layer_2property of the WINDOWCONSTRUCTION object
# window_const.Layer_2 = window_mat_g.Name
# print(f"{window_const}")

# Call out the window shade control that is going to be used
shading_control = idf1.idfobjects["WINDOWSHADINGCONTROL"][0]
print(f"{shading_control}")

# Create a folder with a unique name based on the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
                f"02_Working/Export/{timestamp}"
os.makedirs(output_folder)

for i in range(6):
    window_mat_s = idf1.idfobjects["WINDOWMATERIAL:SHADE"][i]
    print(f"Shade type: {window_mat_s.Name}")
    shading_control.Shading_Type = "ExteriorShade"
    shading_control.Name = f"control_{window_mat_s.Name}"
    shading_control.Shading_Device_Material_Name = f"{window_mat_s.Name}"
    print(f"Shading control: {shading_control}")

    for j in range(8):
        # Rotating the model at 45 intervals
        building.North_Axis = j * 45

        # Set the path to the output files
        output_prefix = f"output_{i}_{j*45}"

        # Make eppy work as EPLaunch in order to get the CSV files.
        # Definition from Eppy Package tutorial
        def make_eplaunch_options(idf1):
            idfversion = idf1.idfobjects['version'][0].Version_Identifier.split('.')
            idfversion.extend([0] * (3 - len(idfversion)))
            idfversionstr = '-'.join([str(item) for item in idfversion])
            fname = idf1.idfname
            options = {
            # 'ep_version':idfversionstr, # runIDFs needs the version number
            # idf.run does not need the above arg
            # you can leave it there and it will be fine :-)
            'output_prefix': output_prefix,
            'output_suffix': 'C',
            'output_directory': output_folder,
            'readvars': True,
            'expandobjects': True
            }
            return options

        # Run the simulation and save to the output files
        theoptions = make_eplaunch_options(idf1)
        idf1.run(**theoptions)


for i in range(6):
    window_mat_b = idf1.idfobjects["WINDOWMATERIAL:BLIND"][i]
    print(f"Shade type: {window_mat_b.Name}")
    shading_control.Shading_Type = "ExteriorBlind"
    shading_control.Name = f"control_{window_mat_b.Name}"
    shading_control.Shading_Device_Material_Name = f"{window_mat_b.Name}"
    print(f"Shading control: {shading_control}")

    for j in range(8):
        # Rotating the model at 45 intervals
        building.North_Axis = j * 45

        # Set the path to the output files
        output_prefix = f"output_{i+6}_{j*45}"

        # Make eppy work as EPLaunch in order to get the CSV files.
        # Definition from Eppy Package tutorial
        def make_eplaunch_options(idf1):
            idfversion = idf1.idfobjects['version'][0].Version_Identifier.split('.')
            idfversion.extend([0] * (3 - len(idfversion)))
            idfversionstr = '-'.join([str(item) for item in idfversion])
            fname = idf1.idfname
            options = {
            # 'ep_version':idfversionstr, # runIDFs needs the version number
            # idf.run does not need the above arg
            # you can leave it there and it will be fine :-)
            'output_prefix': output_prefix,
            'output_suffix': 'C',
            'output_directory': output_folder,
            'readvars': True,
            'expandobjects': True
            }
            return options

        # Run the simulation and save to the output files
        theoptions = make_eplaunch_options(idf1)
        idf1.run(**theoptions)

# Modelling the scenario without shades

