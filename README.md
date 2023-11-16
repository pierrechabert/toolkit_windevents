## Purpose of the toolkit

The purpose of this python toolkit is to create synthetic synoptic intensification and relaxation wind events over a specific region. It was originally designed and created to mimic synoptic wind fluctuations over the southern Senegalese upwelling sector (associated to the publication https://doi.org/10.1175/JPO-D-22-0092.1).
At the end of running this code, you should be having a spatial pattern of anomaly associated to a wind intensification and relaxation over the area of interest, along with gradual visualization of the variables: time series, events detection and spatial patterns. 
Here the variables of interest are meridional and zonal wind stress components, the total net heat fluxes and by construction its decomposition as described in the code.

## Description of the processing
 
First, to download the ERA5 data, use the functions download_dailyERA5_fluxes.py (and download_landmaskERA5.py) in the toolbox editor of the Climate Data Store of Copernicus: https://cds.climate.copernicus.eu/toolbox-editor/. Modify the area, time and variables accordingly to your needs.

Secondly, here is a description of the 9 sections in main_toolkit_windevents.py (that uses functions in fcts_toolkit_windevents.py, and note that the computational python prerequisites/installation are not described as we use commonly used modules):

Section 1) loads the data (and some preprocessing)
Section 2) creates regional averages of variables (modify the region), and plots their time series
Section 3) creates the seasonal cycle of variables
Section 4) filters the seasonal cycle to get only the intraseasonal variability
Section 5) detects and selects the wind events
Section 6) computes the spatial patterns of the variables - absolute and in anomaly - at the selected dates
Section 7) plots the maps of the absolute values during the wind events
Section 8) plots the maps of the anomalies during the wind events
Section 9) proposes a way to export these spatial patterns to netcdf files for further implementation in model grids

## Contact 

If you have any question or comment, please do not hesitate to contact Pierre Chabert at pierre.chabert@locean.ipsl.fr  
