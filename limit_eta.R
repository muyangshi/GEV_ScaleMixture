# Use Likun's code to calculate the limit of eta:
    # T1<-rank(Realizations[,1])/(N+1)
    # T2<-rank(Realizations[,2])/(N+1)
    # NewReal<-data.frame(xi=FrechetInv(T1), xj=FrechetInv(T2))
    # Tmin<-apply(NewReal,1,min)
    # fit<-gpd.fit(Tmin,method='amle')
    # itaX<-fit[1]

library(mvtnorm)
# library(EnvStats)
library(VGAM)
library(extRemes)

# load the CDF_X_{Nt}
load('eta_chi:CDF_X_1000000.gzip')
S1 <- CDF_X[,1]
S2 <- CDF_X[,2]
S3 <- CDF_X[,3]
S4 <- CDF_X[,4]
S5 <- CDF_X[,5]
S6 <- CDF_X[,6]

# transform the CDF_X_{Nt} from uniform to unit Frechet marginal

# weird that using qevd, some Frechet are negative
# S1_Frechet <- qevd(S1,type='Frechet', shape = 1) 
# S2_Frechet <- qevd(S2,type='Frechet', shape = 1)
# S3_Frechet <- qevd(S3,type='Frechet', shape = 1)
# S4_Frechet <- qevd(S4,type='Frechet', shape = 1)
# S5_Frechet <- qevd(S5,type='Frechet', shape = 1)
# S6_Frechet <- qevd(S6,type='Frechet', shape = 1)

# can use the VGAM package's transformation
S1_Frechet_VGAM <- qfrechet(S1, shape = 1)


# using a manual transformation
my_qFrechet <- function(x) {
    return(-1/log(x)) # shape \alpha = 1
}

S1_Frechet_my <- my_qFrechet(S1)
S2_Frechet_my <- my_qFrechet(S2)
S3_Frechet_my <- my_qFrechet(S3)
S4_Frechet_my <- my_qFrechet(S4)
S5_Frechet_my <- my_qFrechet(S5)
S6_Frechet_my <- my_qFrechet(S6)

# Calculate eta's

# eta_12
S12_my <- pmin(S1_Frechet_my, S2_Frechet_my)
S12_fit_my <- fevd(S12_my, threshold=0, type="GP", method="GMLE")
S12_fit_my$results$par
#    scale    shape 
# 1.158062 0.388302 

# eta_34
S34_my <- pmin(S3_Frechet_my, S4_Frechet_my)
S34_fit_my <- fevd(S34_my, threshold=0, type="GP", method="GMLE")
S34_fit_my$results$par
#     scale     shape 
# 1.1512180 0.2750359 

# eta_45
S45_my <- pmin(S4_Frechet_my, S5_Frechet_my)
S45_fit_my <- fevd(S45_my, threshold=0, type="GP", method="GMLE")
S45_fit_my$results$par
#     scale     shape 
# 1.1220280 0.2217238 

# eta_15
S15_my <- pmin(S1_Frechet_my, S5_Frechet_my)
S15_fit_my <- fevd(S15_my, threshold=0, type="GP", method="GMLE")
S15_fit_my$results$par
#     scale     shape 
# 1.1249631 0.1572785 

# eta_36
S36_my <- pmin(S3_Frechet_my, S6_Frechet_my)
S36_fit_my <- fevd(S36_my, threshold=0, type="GP", method="GMLE")
S36_fit_my$results$par
#     scale     shape 
# 1.1266590 0.1560476 

# eta_14
S14_my <- pmin(S1_Frechet_my, S4_Frechet_my)
S14_fit_my <- fevd(S14_my, threshold=0, type="GP", method="GMLE")
S14_fit_my$results$par
#     scale     shape 
# 1.1270437 0.1557162 







