import cdstoolbox as ct

@ct.application(title='Download data')
@ct.output.download()
def download_application():
    data = ct.catalogue.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['land_sea_mask', 
            ],
            'year': [
                '2000',
            ],
            'month': [
                '01',
            ],
            'day': [
                '01', 
            ],
            'time': [
                '12:00', 
            ],
            'area': [
                39, -37.5, 3.75,
                -3.75,
            ],
        }
    )
    
    return data