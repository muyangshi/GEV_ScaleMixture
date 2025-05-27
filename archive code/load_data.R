load("./JJA_precip_maxima.RData")
## ----------------------------------- Descriptions --------------------------------------
## You should see four variables
## -- GEV_estimates: pointwise GEV estimates with a time trend in the location parameter
## -- stations: the (lon, lat) coordinates for 1034 GHCN stations in central United States
## -- JJA_maxima: the summertime (June, July, August) maxima from 1950 - 2017 (68 years in total)
## -- elev: the elevations of the 1034 GHCN stations


## ----------------------------------- Visualization --------------------------------------
library(ggplot2)
library(ggh4x)


## Visualize mu0 estimates
cols <- RColorBrewer::brewer.pal(11, "Spectral")
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=GEV_estimates$mu0))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(mu[0])) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))

## Visualize mu1 estimates
cols <- rev(RColorBrewer::brewer.pal(11, "RdBu"))
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=GEV_estimates$mu1))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(mu[1])) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))


## ---- Visualize elevations of stations ----
ggplot(stations) + 
  geom_point(aes(x=longitude, y=latitude, color=elev))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(elev(s))) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))

## ---- Visualize mu = mu0 + mu1 * time estimates ----
time <- (1950:2017 - mean(1950:2017))/sd(1950:2017) # standardize the years
n.t <- length(time); n.s <- nrow(stations)
Mu0 <- matrix(rep(GEV_estimates$mu0, n.t), ncol=n.t)
Mu1 <- matrix(rep(GEV_estimates$mu1, n.t), ncol=n.t)
Time <- matrix(rep(time, each = n.s), ncol=n.t)
Mu <- Mu0 + Mu1*Time # matrix of size n.s x n.t

## mu in 1950
cols <- rev(RColorBrewer::brewer.pal(11, "RdBu"))
year <- 1950 + 67
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=Mu[,year-1949]))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expr(paste(mu, " in ", !!year))) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))


## Visualize logsigma estimates
cols <- rev(RColorBrewer::brewer.pal(11, "RdBu"))
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=GEV_estimates$logsigma))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(log(sigma))) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))

## Visualize xi estimates
cols <- rev(RColorBrewer::brewer.pal(11, "RdBu"))
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=GEV_estimates$xi))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(xi)) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))



## ------------------ Initial values for thin plates splines coefficients --------------------
k  <- 100

Mu0_coords <- cbind(stations, mu0 = GEV_estimates$mu0)
out <- mgcv::jagam(mu0 ~ elev + s(longitude, latitude, k = k), data=Mu0_coords, diagonalize =  TRUE, centred = TRUE, 
                   na.action= na.pass, file = 'blank.jags1')
splineBasis <- out$jags.data$X
beta_loc0<- out$jags.ini$b
loc0_init_mean <- drop(out$jags.data$X%*%out$jags.ini$b)
sigma_loc0 <- var(loc0_init_mean-GEV_estimates$mu0)
sbeta_loc0 <- 1/out$jags.ini$lambda[1]

cols <- RColorBrewer::brewer.pal(11, "Spectral")
ggplot(stations)+ geom_point(aes(x=longitude, y=latitude, color=loc0_init_mean))+
  geom_polygon( data = map_data("state"), aes(x = long, y = lat, group = group),
                color = 'gray', fill = NA, linewidth = 0.5) +
  scale_color_gradientn(colours = cols, name = expression(paste("Smooth ", mu[0]))) +
  coord_cartesian(xlim = c(-102, -92.1), ylim =  c(32.5, 45)) +
  force_panelsizes(rows = unit(4.05, "in"),
                   cols = unit(2.55, "in"))




Mu1_coords <- cbind(stations, mu1 = GEV_estimates$mu1)
out <- mgcv::jagam(mu1 ~ elev + s(longitude, latitude, k = k), data=Mu1_coords, diagonalize =  TRUE, centred = TRUE, 
                   na.action= na.pass, file = 'blank.jags1')
splineBasis <- out$jags.data$X
beta_loc1<- out$jags.ini$b
loc1_init_mean <- drop(out$jags.data$X%*%out$jags.ini$b)
sigma_loc1 <- var(loc1_init_mean-GEV_estimates$mu1)
sbeta_loc1 <- 1/out$jags.ini$lambda[1]


logSigma_coords <- cbind(stations, logsigma = GEV_estimates$logsigma)
out <- mgcv::jagam(logsigma ~ elev + s(longitude, latitude, k = k), data=logSigma_coords, diagonalize =  TRUE, centred = TRUE, 
                   na.action= na.pass, file = 'blank.jags1')
splineBasis <- out$jags.data$X
beta_logsigma<- out$jags.ini$b
logsigma_init_mean <- drop(out$jags.data$X%*%out$jags.ini$b)
sigma_logsigma <- var(logsigma_init_mean-GEV_estimates$logsigma)
sbeta_logsigma <- 1/out$jags.ini$lambda[1]


Xi_coords <- cbind(stations, xi = GEV_estimates$xi)
out <- mgcv::jagam(xi ~ elev + s(longitude, latitude, k = k), data=Xi_coords, diagonalize =  TRUE, centred = TRUE, 
                   na.action= na.pass, file = 'blank.jags1')
splineBasis <- out$jags.data$X
beta_xi<- out$jags.ini$b
xi_init_mean <- drop(out$jags.data$X%*%out$jags.ini$b)
sigma_xi <- var(xi_init_mean-GEV_estimates$xi)
sbeta_xi <- 1/out$jags.ini$lambda[1]


library(geoR)
plot(variog(coords = stations, data = JJA_maxima[,68]))
