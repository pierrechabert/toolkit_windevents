import cdstoolbox as ct

@ct.application(title='Download data')
@ct.output.download()
def download_application():
    data = ct.catalogue.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                  '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_surface_downward_long_wave_radiation_flux',
                'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux',
                'mean_surface_net_short_wave_radiation_flux', 'mean_surface_sensible_heat_flux', 'total_cloud_cover',
            ],
            'year': [
                '2000',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '04:00', '08:00',
                '12:00', '16:00', '20:00',
            ],
            'area': [
                39, -37.5, 3.75,
                -3.75,
            ],
        }
    )
    
    data_daily = ct.climate.daily_mean(data)
    
    return data_daily
