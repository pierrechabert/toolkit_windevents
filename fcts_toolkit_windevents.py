import numpy as np
import xarray as xr 
import cmocean
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import scipy.signal
import matplotlib

def monthly_avg(arr, timedates):

	l, m, n = arr.shape
	arrmean = np.zeros((12, m, n))
	cntr = np.zeros((12, m, n))

	for i in range(l):
		idx = timedates[i].month - 1
		arrmean[idx] += arr[i] 
		cntr[idx] += 1

	arrmean = arrmean / cntr

	return(arrmean)

def yearly_avg(arr, timedates):

	l, m, n = arr.shape
	nmdy = 365
	nmy = l // nmdy
	arrseas = np.zeros((nmdy, m, n))
	dly = 0
	for i in range(nmy):
		for j in range(nmdy):
			if timedates[nmdy*i + j + dly].month==2 and timedates[nmdy*i + j + dly].day==29:
				dly += 1
			arrseas[j,:,:] += arr[nmdy*i + j + dly,:,:]	

	arrseas = arrseas/nmy

	return(arrseas)

def nans_outbox(array, lon, lat, lonmin, lonmax, latmin, latmax):

	if len(array.shape) <= 2:
		array[lon>lonmax] = np.nan
		array[lon<lonmin] = np.nan
		array[lat>latmax] = np.nan
		array[lat<latmin] = np.nan	
	else:
		for i in range(array.shape[0]):
			array[i][lon>lonmax] = np.nan
			array[i][lon<lonmin] = np.nan
			array[i][lat>latmax] = np.nan
			array[i][lat<latmin] = np.nan
	return(array)


def typical_pattern_events(arr, idxs):

	typical_arr = np.zeros(arr[0].shape)
	for i in range(len(idxs)):
		typical_arr = np.dstack((typical_arr, arr[idxs[i]]))

	typical_arr = typical_arr[:,:,1:]
	typical_arr = np.nanmean(typical_arr, axis=2)

	return(typical_arr)

# for each events detected (each date), the anomaly is computed (for each day: event - climato)
# then we average and obtain the typical anomaly
def typical_pattern_events_anomaly(arr, arrseas, timedates, idxs):

	typical_arr = np.zeros(arr[0].shape)
	for i in range(len(idxs)):
		if arrseas.shape[0]==365:
			idxdy = timedates[idxs[i]].timetuple().tm_yday # computes the day of the year to remove the corresponding climatological spatial average
		if arrseas.shape[0]==12:
			idxdy = timedates[idxs[i]].month - 1
		typical_arr = np.dstack((typical_arr, arr[idxs[i]] - arrseas[idxdy]))

	typical_arr = typical_arr[:,:,1:]
	typical_arr = np.nanmean(typical_arr, axis=2)

	return(typical_arr)


def export_nc(arr_exp, lon_ar, lat_ar, filename, varstr, descriptionstr):

    dataset = Dataset(filename , 'w')
    dataset.description = descriptionstr

    #dataset.history = 'Created ' + time.ctime(time.time())
    dataset.source = 'P. Chabert (LOCEAN-IPSL)'
    lat = dataset.createDimension('time', 1)
    lat = dataset.createDimension('lat', arr_exp[:,0].size)
    lon = dataset.createDimension('lon', arr_exp[0,:].size)

    latitudes = dataset.createVariable('time', np.float32, ('time',))
    latitudes = dataset.createVariable('latitude', np.float32, ('lat',))
    longitudes = dataset.createVariable('longitude', np.float32, ('lon',))
    data = dataset.createVariable(varstr, np.float32, ('time', 'lat', 'lon'))

    latitudes[:] = lat_ar[:,0]
    longitudes[:] = lon_ar[0,:]

    data[:] = arr_exp

    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    data.units = ' '

    dataset.close()


