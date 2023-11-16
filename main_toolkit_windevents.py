import numpy as np
import xarray as xr 
import cmocean
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import scipy.signal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import fcts_toolkit_windevents

################################################################################
## Detection and creation of intensification and relaxation wind events
## in wind stress and heat flux 
################################################################################

################################################################################
##  1) data loading
# lathf = flux de chaleur latente = surface latent heat flux
# senhf = flux sensible = surface sensible heat flux
# swrad = flux solaire net = downward surface solar radiation
# lwrad = flux infrarouge downward = downward surface thermal radiation
# lwradup = flux infrarouge upward = (upward) surface thermal radiation
# U, V = vitesse à 10 metres = wind speed at 10 meters (instantaneous)

# http://oaflux.whoi.edu/descriptionheatflux.html
# Net heat flux Qnet is computed as:

#       Qnet = SW - LW - LH - SH.

# where 
# SW denotes net downward shortwave radiation
# LW net upward longwave radiation
# LH latent heat flux
# SH sensible heat flux
# unit is W/m2.

datapath = '/net/ether/data/proteo1/pchabert/mod/IRENE_2020/ERA5_2000-2010_CAN/'
savepath = '/net/ether/data/proteo1/pchabert/mod/IRENE_2020/run1test/CAN11_NEW/post_processing/heatfluxes/arrays_events/'

fnmprefix = 'ERA5_2000-2010_CAN_'

globals()['ds_u10'] = xr.open_dataset(datapath + fnmprefix + 'u10.nc')
globals()['ds_v10'] = xr.open_dataset(datapath + fnmprefix + 'v10.nc')
globals()['ds_slathf'] = xr.open_dataset(datapath + fnmprefix + 'slathf.nc')
globals()['ds_ssenhf'] = xr.open_dataset(datapath + fnmprefix + 'ssenhf.nc')
globals()['ds_swrad'] = xr.open_dataset(datapath + fnmprefix + 'swrad.nc')
globals()['ds_lwrad'] = xr.open_dataset(datapath + fnmprefix + 'lwrad.nc')
globals()['ds_lwradfdownwards'] = xr.open_dataset(datapath + fnmprefix + 'lwradfdownwards.nc')
globals()['ds_swradfdownwards'] = xr.open_dataset(datapath + fnmprefix + 'swradfdownwards.nc')
globals()['ds_totalcloudcoverage'] = xr.open_dataset(datapath + fnmprefix + 'totalcloudcoverage.nc')

ds_mask = xr.open_dataset(datapath + fnmprefix + 'landmask.nc')
maskERA5 = ds_mask.sftlf.values[0][0]

lon0 = ds_slathf.lon.values
lat0 = ds_slathf.lat.values
lon, lat = np.meshgrid(lon0, lat0)
time = ds_slathf.time.values
timenum = matplotlib.dates.date2num(time)
timenum0 = timenum - timenum[0]
timedates = matplotlib.dates.num2date(timenum)
timedates = np.array(timedates)

U = ds_u10.uas.values[0]
V = ds_v10.vas.values[0]
SW = ds_swradfdownwards.msdwswrf_NON_CDM.values[0] # downward short wave radiation flux
LW = ds_lwrad.msnlwrf_NON_CDM.values[0] # upward long wave radiation flux (mean surface net long wave radiation flux)
LH = ds_slathf.mslhf_NON_CDM.values[0] # latent heat flux
SH = ds_ssenhf.msshf_NON_CDM.values[0] # sensible heat flux

TCC = ds_totalcloudcoverage.clt.values[0]

# removes bisextil days (to have only 365 day years) 
# and removes last year, to have exatcly the same days as QSCAT
idxs_bise = []
for i in range(timedates.shape[0]):
	if timedates[i].month==2 and timedates[i].day==29:
		idxs_bise.append(i)

U = np.delete(U, idxs_bise, axis=0)[:-365]
V = np.delete(V, idxs_bise, axis=0)[:-365]
SW = np.delete(SW, idxs_bise, axis=0)[:-365]
LW = np.delete(LW, idxs_bise, axis=0)[:-365]
SH = np.delete(SH, idxs_bise, axis=0)[:-365]
LH = np.delete(LH, idxs_bise, axis=0)[:-365]
TCC = np.delete(TCC, idxs_bise, axis=0)[:-365]
time = np.delete(time, idxs_bise, axis=0)[:-365]
timenum = np.delete(timenum, idxs_bise, axis=0)[:-365]
timenum0 = np.delete(timenum0, idxs_bise, axis=0)[:-365]
timedates = np.delete(timedates, idxs_bise, axis=0)[:-365]

# computes net heat flux
Qnet = SW + LW + LH + SH 

# https://marine.rutgers.edu/dmcs/ms501/2004/Notes/Wilkin20041014.htm
# computes wind stress from wind speed
rhoa = 1.22
CD = 0.0013
tauU = np.sign(U) * rhoa * CD * U * U
tauV = np.sign(V) * rhoa * CD * V * V


################################################################################
################################################################################
## 2) processing - removal of nans and regional box averages

# region of interest
lonmin, lonmax, latmin, latmax = -19, -16.5, 12.5, 15.5

# monthly avg (clim) + avg in the box on the net heat flux
Qnetclim = fcts_toolkit_windevents.monthly_avg(Qnet, timedates)
Qnetclimbox = fcts_toolkit_windevents.nans_outbox(Qnetclim.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
QnetclimboxTS = np.nanmean(Qnetclimbox, axis=(1,2))

# regional box averages
Ubox = fcts_toolkit_windevents.nans_outbox(U.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
Vbox = fcts_toolkit_windevents.nans_outbox(V.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
tauUbox = fcts_toolkit_windevents.nans_outbox(tauU.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
tauVbox = fcts_toolkit_windevents.nans_outbox(tauV.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
SWbox = fcts_toolkit_windevents.nans_outbox(SW.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
LWbox = fcts_toolkit_windevents.nans_outbox(LW.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
SHbox = fcts_toolkit_windevents.nans_outbox(SH.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
LHbox = fcts_toolkit_windevents.nans_outbox(LH.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
TCCbox = fcts_toolkit_windevents.nans_outbox(TCC.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
Qnetbox = fcts_toolkit_windevents.nans_outbox(Qnet.copy(), lon, lat, lonmin, lonmax, latmin, latmax)

UboxTS = np.nanmean(Ubox, axis=(1,2))
VboxTS = np.nanmean(Vbox, axis=(1,2))
tauUboxTS = np.nanmean(tauUbox, axis=(1,2))
tauVboxTS = np.nanmean(tauVbox, axis=(1,2))
SWboxTS = np.nanmean(SWbox, axis=(1,2))
LWboxTS = np.nanmean(LWbox, axis=(1,2))
SHboxTS = np.nanmean(SHbox, axis=(1,2))
LHboxTS = np.nanmean(LHbox, axis=(1,2))
TCCboxTS = np.nanmean(TCCbox, axis=(1,2))
QnetboxTS = np.nanmean(Qnetbox, axis=(1,2))

arrays = [UboxTS, VboxTS, SWboxTS, LWboxTS, SHboxTS, LHboxTS, TCCboxTS, QnetboxTS]
arrays_str = ['U', 'V', 'SW', 'LW', 'SH', 'LH', 'TCC', 'Qnet']

## to plot the 10 years time series of box-averaged quantities
plt.figure(figsize=(12,16))
for i in range(8):
	plt.subplot(4,2,i+1)
	plt.plot(timedates, arrays[i], c='b', label=arrays_str[i])
	plt.legend()
	plt.grid()
	plt.xlabel('time')
#plt.show()

################################################################################
## 3) creation of the seasonal cycle (keeping the spatial component) for  ERA5 - yearly averages

Useas = fcts_toolkit_windevents.yearly_avg(U, timedates)
Vseas = fcts_toolkit_windevents.yearly_avg(V, timedates)
tauUseas = fcts_toolkit_windevents.yearly_avg(tauU, timedates)
tauVseas = fcts_toolkit_windevents.yearly_avg(tauV, timedates)
SWseas = fcts_toolkit_windevents.yearly_avg(SW, timedates)
LWseas = fcts_toolkit_windevents.yearly_avg(LW, timedates)
SHseas = fcts_toolkit_windevents.yearly_avg(SH, timedates)
LHseas = fcts_toolkit_windevents.yearly_avg(LH, timedates)
Qnetseas = fcts_toolkit_windevents.yearly_avg(Qnet, timedates)

Uclim = fcts_toolkit_windevents.monthly_avg(U, timedates)
Vclim = fcts_toolkit_windevents.monthly_avg(V, timedates)
tauUclim = fcts_toolkit_windevents.monthly_avg(tauU, timedates)
tauVclim = fcts_toolkit_windevents.monthly_avg(tauV, timedates)
SWclim = fcts_toolkit_windevents.monthly_avg(SW, timedates)
LWclim = fcts_toolkit_windevents.monthly_avg(LW, timedates)
SHclim = fcts_toolkit_windevents.monthly_avg(SH, timedates)
LHclim = fcts_toolkit_windevents.monthly_avg(LH, timedates)
TCCclim = fcts_toolkit_windevents.monthly_avg(TCC, timedates)
Qnetclim = fcts_toolkit_windevents.monthly_avg(Qnet, timedates)

tauUseasbox = fcts_toolkit_windevents.nans_outbox(tauUseas.copy(), lon, lat, lonmin, lonmax, latmin, latmax)
tauVseasbox = fcts_toolkit_windevents.nans_outbox(tauVseas.copy(), lon, lat, lonmin, lonmax, latmin, latmax)

tauUseasboxTS = np.nanmean(tauUseasbox, axis=(1,2))
tauVseasboxTS = np.nanmean(tauVseasbox, axis=(1,2))


################################################################################
## 4) filtering of the seasonal cycle signal - we get only the intraseasonal variability

# we compute a high-pass Butterworth filter with a period limit of 115 days
fs = 1/86400 # data everyday
N, Wn = 1, 1e-7 # 1/1e-7/86400 = 115 days
b, a = scipy.signal.butter(N, Wn, 'low', fs=fs)
tauUseasboxTSfilt = scipy.signal.filtfilt(b, a, tauUseasboxTS)
tauVseasboxTSfilt = scipy.signal.filtfilt(b, a, tauVseasboxTS)

tauUseasboxTSfiltT = tauUseasboxTSfilt.copy()
tauVseasboxTSfiltT = tauVseasboxTSfilt.copy()

tauUseasboxTSrafiltT = tauUboxTS.copy()
tauVseasboxTSrafiltT = tauVboxTS.copy()

for i in range(10):
	tauUseasboxTSrafiltT[365*i:365*(i+1)] = tauUseasboxTSrafiltT[365*i:365*(i+1)] - tauUseasboxTSfiltT
	tauVseasboxTSrafiltT[365*i:365*(i+1)] = tauVseasboxTSrafiltT[365*i:365*(i+1)] - tauVseasboxTSfiltT

threshold_intensewind_tauU = -np.ones(tauUseasboxTSrafiltT.shape)*np.nanstd(tauUseasboxTSrafiltT)
threshold_intensewind_tauV = -np.ones(tauVseasboxTSrafiltT.shape)*np.nanstd(tauVseasboxTSrafiltT)

tseas = np.linspace(1, 365, 365)

################################################################################
## 5) selection of typical intense wind events and relaxation events

idxs = []
idxs_upos, idxs_uneg = [], []
idxs_rel = []
idxs_stress = []
idxs_upos_stress, idxs_uneg_stress = [], []
idxs_rel_stress = []
idxmask_upwelseas = [150, 300, 365+150, 365+300, 365*2+150, 365*2+300, 365*3+150, 365*3+300, 365*4+150, 365*4+300, 365*5+150, 365*5+300, 365*6+150, 365*6+300, 365*7+150, 365*7+300, 365*8+150, 365*8+300, 365*9+150, 365*9+300, 365*10+150, 365*10+300] # mask non-upwelling season (upwelling season is from day 300 to day 150 the next year)

time_mask = np.ones(time.shape)
# mask non-upwelling seasons
for i in range(0, len(idxmask_upwelseas), 2):
	time_mask[idxmask_upwelseas[i]:idxmask_upwelseas[i+1]] = 0

# select events based on wind stress criteria
for i in range(tauUseasboxTSrafiltT.shape[0]-1):

	critV = tauVseasboxTSrafiltT
	critVrel = tauVboxTS
	thresV = threshold_intensewind_tauV[0]
	thresVrel = -threshold_intensewind_tauV[0] # wind stress > std = relaxation

	# criteria on the std + local extrema + upwelling season selection
	if critV[i] < thresV and critV[i] < critV[i-1] and critV[i] < critV[i+1] and time_mask[i]==1:
		idxs_stress.append(i)

	if critV[i] > thresVrel and critV[i] > critV[i-1] and critV[i] > critV[i+1] and time_mask[i]==1:
		idxs_rel_stress.append(i)

# plot the time series with events detection
plt.figure(figsize=(8,5))
plt.subplot(111); ax = plt.gca()
ax.plot(timenum0, tauVboxTS, c='g', label='Box avg.')
ax.plot(np.hstack((tseas, tseas+365, tseas+2*365, tseas+3*365, tseas+4*365, tseas+5*365, tseas+6*365, tseas+7*365, tseas+8*365, tseas+9*365)), np.hstack((tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT, tauVseasboxTSfiltT)), label='Seas. cycle', c='b')
ax.plot(timenum0, tauVseasboxTSrafiltT, c='r', label='Box avg. - seas. cycle')
plt.plot(timenum0, threshold_intensewind_tauV, c='grey')
plt.plot(timenum0, -threshold_intensewind_tauV, c='grey')
plt.legend(); plt.grid()
plt.xlabel('Time (days)');ax.set_ylabel('Meridional wind stress ($N.m^{-2}$)')
maxx = np.nanmax(np.abs(tauVboxTS))
for j in range(len(idxs_stress)):
	plt.plot(timenum0[idxs_stress[j]], tauVseasboxTSrafiltT[idxs_stress[j]], '*', c='yellow')
for j in range(len(idxs_rel_stress)):
	plt.plot(timenum0[idxs_rel_stress[j]], tauVseasboxTSrafiltT[idxs_rel_stress[j]], '*', c='orange')
#plt.show()



################################################################################
## 6) computes the spatial patterns of wind stress and net heat flux both in absolute and anomaly
# the spatial average is made on large scale but criteria (idxs and idxs_rel) of wind event on reduced area (box avg)

# construction of intense wind events wind and fluxes patterns wind stress criteria
typical_intense_U_stress = fcts_toolkit_windevents.typical_pattern_events(tauU, idxs_stress)
typical_intense_V_stress = fcts_toolkit_windevents.typical_pattern_events(tauV, idxs_stress)
typical_intense_SW_stress = fcts_toolkit_windevents.typical_pattern_events(SW, idxs_stress)
typical_intense_LW_stress = fcts_toolkit_windevents.typical_pattern_events(LW, idxs_stress)
typical_intense_SH_stress = fcts_toolkit_windevents.typical_pattern_events(SH, idxs_stress)
typical_intense_LH_stress = fcts_toolkit_windevents.typical_pattern_events(LH, idxs_stress)
typical_intense_TCC_stress = fcts_toolkit_windevents.typical_pattern_events(TCC, idxs_stress)
typical_intense_Qnet_stress = fcts_toolkit_windevents.typical_pattern_events(Qnet, idxs_stress)

# construction of relaxation wind events wind and fluxes patterns wind stress criteria
typical_rel_U_stress = fcts_toolkit_windevents.typical_pattern_events(tauU, idxs_rel_stress)
typical_rel_V_stress = fcts_toolkit_windevents.typical_pattern_events(tauV, idxs_rel_stress)
typical_rel_SW_stress = fcts_toolkit_windevents.typical_pattern_events(SW, idxs_rel_stress)
typical_rel_LW_stress = fcts_toolkit_windevents.typical_pattern_events(LW, idxs_rel_stress)
typical_rel_SH_stress = fcts_toolkit_windevents.typical_pattern_events(SH, idxs_rel_stress)
typical_rel_LH_stress = fcts_toolkit_windevents.typical_pattern_events(LH, idxs_rel_stress)
typical_rel_TCC_stress = fcts_toolkit_windevents.typical_pattern_events(TCC, idxs_rel_stress)
typical_rel_Qnet_stress = fcts_toolkit_windevents.typical_pattern_events(Qnet, idxs_rel_stress)

# construction of intense wind events wind and fluxes patterns anomalies _stress
typical_intense_U_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(tauU, tauUclim, timedates, idxs_stress)
typical_intense_V_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(tauV, tauVclim, timedates, idxs_stress)
typical_intense_SW_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(SW, SWclim, timedates, idxs_stress)
typical_intense_LW_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(LW, LWclim, timedates, idxs_stress)
typical_intense_SH_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(SH, SHclim, timedates, idxs_stress)
typical_intense_LH_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(LH, LHclim, timedates, idxs_stress)
typical_intense_TCC_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(TCC, TCCclim, timedates, idxs_stress)
typical_intense_Qnet_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(Qnet, Qnetclim, timedates, idxs_stress)

# construction of relaxation wind events wind and fluxes patterns anomalies _stress
typical_rel_U_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(tauU, tauUclim, timedates, idxs_rel_stress)
typical_rel_V_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(tauV, tauVclim, timedates, idxs_rel_stress)
typical_rel_SW_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(SW, SWclim, timedates, idxs_rel_stress)
typical_rel_LW_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(LW, LWclim, timedates, idxs_rel_stress)
typical_rel_SH_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(SH, SHclim, timedates, idxs_rel_stress)
typical_rel_LH_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(LH, LHclim, timedates, idxs_rel_stress)
typical_rel_TCC_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(TCC, TCCclim, timedates, idxs_rel_stress)
typical_rel_Qnet_anomaly_stress = fcts_toolkit_windevents.typical_pattern_events_anomaly(Qnet, Qnetclim, timedates, idxs_rel_stress)


################################################################################
## 7) plots maps of absolute values during events
cmapheat = cmocean.cm.thermal
cmapheat = 'jet'
valminmax = .15
clabU = 'tauU (N/m2)'
clabV = 'tauV (N/m2)'
valminSW, valmaxSW = 150, 250
valminLW, valmaxLW = -150, 0
valminSH, valmaxSH = -50, 0
valminLH, valmaxLH = -220, 0
valminTCC, valmaxTCC = 0, 75
valminQnet, valmaxQnet = -50, 150

# intense wind event
plt.figure(figsize=(14,5)), plt.suptitle('typical intense wind event (absolute)')
plt.subplot(241, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_U_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabU)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(242, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_V_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabV)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(243, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_SW_stress, vmin=valminSW, vmax=valmaxSW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(244, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_LW_stress, vmin=valminLW, vmax=valmaxLW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(245, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_SH_stress, vmin=valminSH, vmax=valmaxSH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(246, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_LH_stress, vmin=valminLH, vmax=valmaxLH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(247, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_TCC_stress, vmin=valminTCC, vmax=valmaxTCC, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('TCC (%)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(248, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_Qnet_stress, vmin=valminQnet, vmax=valmaxQnet, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('Qnet (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')


# rel wind stress
plt.figure(figsize=(14,5)), plt.suptitle('typical relaxation wind event (absolute)')
plt.subplot(241, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_U_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabU)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(242, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_V_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabV)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(243, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_SW_stress, vmin=valminSW, vmax=valmaxSW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(244, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_LW_stress, vmin=valminLW, vmax=valmaxLW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(245, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_SH_stress, vmin=valminSH, vmax=valmaxSH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(246, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_LH_stress, vmin=valminLH, vmax=valmaxLH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(247, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_TCC_stress, vmin=valminTCC, vmax=valmaxTCC, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('TCC (%)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(248, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_Qnet_stress, vmin=valminQnet, vmax=valmaxQnet, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('Qnet (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
#plt.show()


################################################################################
## 8) plots maps of anomalous values during events
cmapheat = cmocean.cm.balance
valminmax = -.05
clabU = 'tauU (N/m2)'
clabV = 'tauV (N/m2)'
valSW = 50
valLW = 50
valSH = 50
valLH = 50
valTCC = 30
valQnet = 50

# intense wind event
plt.figure(figsize=(14,5)), plt.suptitle('typical intense wind event anomaly')
plt.subplot(241, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_U_anomaly_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabU)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(242, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_V_anomaly_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabV)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(243, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_SW_anomaly_stress, vmin=-valSW, vmax=valSW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(244, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_LW_anomaly_stress, vmin=-valLW, vmax=valLW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(245, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_SH_anomaly_stress, vmin=-valSH, vmax=valSH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(246, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_LH_anomaly_stress, vmin=-valLH, vmax=valLH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(247, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_TCC_anomaly_stress, vmin=-valTCC, vmax=valTCC, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('TCC (%)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(248, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_intense_Qnet_anomaly_stress, vmin=-valQnet, vmax=valQnet, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('Qnet (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')

# relaxation wind event
plt.figure(figsize=(14,5)), plt.suptitle('typical relaxation wind event anomaly')
plt.subplot(241, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_U_anomaly_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabU)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(242, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_V_anomaly_stress, vmin=-valminmax, vmax=valminmax, cmap=cmocean.cm.balance)
cb = plt.colorbar(); cb.set_label(clabV)
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(243, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_SW_anomaly_stress, vmin=-valSW, vmax=valSW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(244, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_LW_anomaly_stress, vmin=-valLW, vmax=valLW, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LW (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(245, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_SH_anomaly_stress, vmin=-valSH, vmax=valSH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('SH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(246, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_LH_anomaly_stress, vmin=-valLH, vmax=valLH, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('LH (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(247, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_TCC_anomaly_stress, vmin=-valTCC, vmax=valTCC, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('TCC (%)')
ax = plt.gca()
ax.coastlines(resolution='10m')
plt.subplot(248, projection=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, typical_rel_Qnet_anomaly_stress, vmin=-valQnet, vmax=valQnet, cmap=cmapheat)
cb = plt.colorbar(); cb.set_label('Qnet (W/m2)')
ax = plt.gca()
ax.coastlines(resolution='10m')

#plt.show()



################################################################################
## 9) export anomalies to process them and inject them into CROCO
# only intensification as it is considered symmetric with relaxation
export_anomalies = False

if export_anomalies:

	# mask land before netcdf export
	landsea_threshold = 0 # or 0.1, 0.3 -> fraction of land on the pixel # here strict criteria

	typical_intense_Qnet_anomaly_stress[maskERA5>landsea_threshold] = np.nan
	typical_intense_U_anomaly_stress[maskERA5>landsea_threshold] = np.nan
	typical_intense_V_anomaly_stress[maskERA5>landsea_threshold] = np.nan

	# export as netcdf ERA5 anomalies of Qnet, susvtr, svstr: intensification
	arr_exp = typical_intense_Qnet_anomaly_stress
	lon_ar, lat_ar = lon, lat
	filename = savepath + 'typical_intense_Qnet_anomaly_landmask0.nc'
	varstr = 'Qnet'
	descriptionstr = 'Qnet for typical intensification wind event, constructed from ERA5 on SSUS criteria'
	fcts_toolkit_windevents.export_nc(arr_exp, lon_ar, lat_ar, filename, varstr, descriptionstr)

	arr_exp = typical_intense_U_anomaly_stress
	lon_ar, lat_ar = lon, lat
	filename = savepath + 'typical_intense_sustr_anomaly_landmask0.nc'
	varstr = 'sustr'
	descriptionstr = 'surface zonal wind stress for typical intensification wind event, constructed from ERA5 on SSUS criteria'
	fcts_toolkit_windevents.export_nc(arr_exp, lon_ar, lat_ar, filename, varstr, descriptionstr)

	arr_exp = typical_intense_V_anomaly_stress
	lon_ar, lat_ar = lon, lat
	filename = savepath + 'typical_intense_svstr_anomaly_landmask0.nc'
	varstr = 'svstr'
	descriptionstr = 'surface meridional wind stress for typical intensification wind event, constructed from ERA5 on SSUS criteria'
	fcts_toolkit_windevents.export_nc(arr_exp, lon_ar, lat_ar, filename, varstr, descriptionstr)

################################################################################
