from libs.modules.my_methods import readNcSmapToDf
import datetime

lat_point = -35.5
lon_point = 174.0
start_date = datetime.datetime(2016, 6, 1)
end_date = datetime.datetime(2016, 6, 30)  # year, month, day
data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'

readNcSmapToDf(lat_point, lon_point, start_date, end_date, data_path)

# data_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\TopnetFiles\'
# data_filename = r'streamq_daily_average_1974010100_2019062900_utc_topnet_Northlan_strahler3-NM.nc'
# readNcTopnet(lat_point, lon_point, start_date, end_date, data_path)
