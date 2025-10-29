## Setup from Scratch

#folder structure
dir.create("_data")
dir.create("_Rscripts")
dir.create("_plot_images")
dir.create("_tables")
#dir.create("_cache")  #uncomment if you want to use caching; requires additional setup, see README.md

#can delete after use, or drop/save in _Rscripts
file.copy("0_setup_from_scratch.r", "_Rscripts/0_setup_from_scratch.r")
file.copy("../data_sources/Crashes Data 2019-2023.csv", "_data/Crashes_Data_2019-2023.csv") #some example data
