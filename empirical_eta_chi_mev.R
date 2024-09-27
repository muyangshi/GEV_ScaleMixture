# Need the pRW of X_star, i.e., transformed to uniform
load('eta_chi:X_star.gzip')

library(mev)

X_star <- t(X_star)
est <- taildep(data = X_star[,1:2],
        u = c(0.9,0.999),
        depmeas = 'eta')
