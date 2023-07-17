import datetime
from eppy.modeleditor import IDF
import os

# Set the path to the EnergyPlus executable
eplus_path = "C:/EnergyPlusV22-2-0/EnergyPlus.exe"

# Set the path to the IDD file
iddfile = "C:/EnergyPlusV22-2-0/Energy+.idd"

# Load the IDF file
idf_file = "C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
           "02_Working/EnergyPlus/20230419_T8_ep_file/Forloop_noshade.idf"
epw_file = "C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
           "02_Working/epw/NLD_Amsterdam.062400_IWEC/NLD_Amsterdam.062400_IWEC.epw"

IDF.setiddname(iddfile)
idf1 = IDF(idf_file, epw_file)

building = idf1.idfobjects['BUILDING'][0]
print(building.Name)

# Create a folder with a unique name based on the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/" \
                f"02_Working/Export/{timestamp}"
os.makedirs(output_folder)

# Modelling the scenario without shades

for j in range(8):
    # Rotating the model at 45 intervals
    building.North_Axis = j * 45

    # Set the path to the output files
    output_prefix = f"output_noshade_{j * 45}"

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