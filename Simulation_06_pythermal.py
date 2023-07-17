from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
import pandas as pd

df_file = pd.read_excel(f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                        f"02_Working/Export/20230419_113656/00_temp.xlsx")

# create new DataFrames to store pmv and ppd values
pmv_df = pd.DataFrame(columns=df_file.columns)
ppd_df = pd.DataFrame(columns=df_file.columns)

# define input variable parameters
rh = 50
v = 0.1
met = 1.4
clo = 0.5

# calculate relative air speed
v_r = v_relative(v=v, met=met)

# calculate dynamic clothing
clo_d = clo_dynamic(clo=clo, met=met)

# loop over the columns
for col in df_file.columns:

    # get the values for the column
    operative_temperatures = df_file[col]

    # calculate the PMV and PPD for each row
    pmv_values = []
    ppd_values = []

    for tdb in operative_temperatures:

        #tr is assumed to be the same as tdb
        tr = tdb

        #conduct pmv-ppd analysis
        results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d, standard='ASHRAE',)

        pmv_values.append(results['pmv'])
        ppd_values.append(results['ppd'])

        print(pmv_values)

    print(len(pmv_values), len(ppd_values))

    pmv_df[col] = pd.Series(pmv_values)
    ppd_df[col] = pd.Series(ppd_values)

print(pmv_df)

pmv_df.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                     f"02_Working/Export/20230419_113656/00_pmv.xlsx"), header=True, index=False)

ppd_df.to_excel((f"C:/Users/prana/OneDrive - Delft University of Technology/Thesis/"
                     f"02_Working/Export/20230419_113656/00_ppd.xlsx"), header=True, index=False)