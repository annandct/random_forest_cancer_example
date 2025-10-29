.default_slick_knit_ops <- function(
  #default arguments
  verbose=TRUE,
  base_dir = "../", 
  base_url = "",
  fig_path = "../_plot_images/", 
  cache_path = "../_cache/", 
  #knit arguments
  message = FALSE, 
  warning = FALSE, 
  cache = FALSE) {
  knitr::opts_chunk$set(fig.path = fig_path,
                        cache.path = cache_path,  
                        message = message, 
                        warning = warning,
                        cache = cache)
  if(verbose==TRUE){
    print(cat(paste0("Base Directory: ", base_dir, "\n",
                    "Base URL: ", base_url, "\n",
                    "Figure Path: ", fig_path, "\n",
                    "Cache Path: ", cache_path, "\n",
                    "Message: ", message, "\n",
                    "Warning: ", warning, "\n",
                    "Cache: ", cache, "\n")))
  }
}

# Example usage:
#.base_dir <- "../" 
#.base_url <- "" # keep as is - only needed if for html output/github pages versions (see readme and other templates
#fig_path <- "../_plot_images/" # where image output should go

#knitr::opts_knit$set(base.dir = .base_dir) #paste0(.base_url, .base_dir)
# if using github pages output - you will want to add you GITHUB PAGES URL as the base URL
#knitr::opts_chunk$set(fig.path = .fig_path,

#cache.path = '../_cache/',  #log files go here - IF YOU ARE USING CACHE
# Some reproducible examples will want this FALSE/EXLCUDED so there isn't any 'hidden (cached) data' making it's way into your analysis/scripts. 
                      
#knit ops... 
#message=FALSE, warning=FALSE, cache = FALSE ## set to TRUE if you want to use caching