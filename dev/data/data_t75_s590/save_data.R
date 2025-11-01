load("./JJA_precip_maxima_imputed.RData")
library(readr)
blockMax_JJA_centralUS <- read_csv("blockMax_JJA_centralUS.csv", 
                                   col_names = FALSE, skip = 1)

JJA_maxima_nonimputed <- as.matrix(blockMax_JJA_centralUS)

save(GEV_estimates,
     JJA_maxima_nonimputed,
     stations,
     elev,
     file = "JJA_precip_maxima_nonimputed.RData")


load("./JJA_precip_maxima_nonimputed.RData")
