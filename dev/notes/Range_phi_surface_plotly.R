## This document is very important because:
## -- 1. It can specify the size of a plotly figure;
## -- 2. Save the snapshots with desired camera eye;
## -- 3. Trim the wide margins with magick.


weights_fun<-function(d,radius,h=1, cutoff=TRUE){
  tmp = exp(-d**2/(2*h))
  if(cutoff) tmp[d>radius] = 0
  
  return(tmp/sum(tmp))
}
  

setwd('~/Desktop/')
loc_tmp=seq(0,10,length.out = 7)[c(2,4,6)]
Knots_data = expand.grid(x=loc_tmp, y=loc_tmp)

## --------------------------------------------------------------------------
## ----------------------------- range surface ------------------------------
## --------------------------------------------------------------------------
### ----  Scenario 1 ----
# range_at_knots = (-1 + 0.3*Knots_data[,1] + 0.4*Knots_data[,2])/4
range_at_knots = sqrt(0.3*Knots_data[,1] + 0.4*Knots_data[,2])/2
bw = 4 # bandwidth
n_s=25*25
station_tmp = seq(0,10,length.out = 25)
Stations = expand.grid(x=station_tmp,y=station_tmp)

range_vec = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  range_vec[idx] = sum(weights*range_at_knots)
}
  
library(plotly)
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(range_vec,nrow=length(station_tmp)), colors='#196cb5', showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(range_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03C1(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
# setwd('~/Desktop/')
orca(fig, file='./PyCode/Simulation_figures/range_scenario1.png')

library(magick)
img <- image_read('~/Desktop/PyCode/Simulation_figures/range_scenario1.png')
img <- image_trim(img)
print(img)
image_write(img, path='~/Desktop/PyCode/Simulation_figures/range_scenario1_trimmed.png', format='png')

### ----  Scenario 2 ----
range_at_knots = sqrt(0.3*Knots_data[,1] + 0.4*Knots_data[,2])/2
bw = 4 # bandwidth
n_s=25*25
station_tmp = seq(0,10,length.out = 25)
Stations = expand.grid(x=station_tmp,y=station_tmp)

range_vec = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  range_vec[idx] = sum(weights*range_at_knots)
}

library(plotly)
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(range_vec,nrow=length(station_tmp)), colors='#196cb5', showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(range_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03C1(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
# setwd('~/Desktop/')
orca(fig, file='./PyCode/Simulation_figures/range_scenario1.png')

library(magick)
img <- image_read('~/Desktop/PyCode/Simulation_figures/range_scenario1.png')
img <- image_trim(img)
print(img)
image_write(img, path='~/Desktop/PyCode/Simulation_figures/range_scenario1_trimmed.png', format='png')




## --------------------------------------------------------------------------
## ------------------------------ phi surface -------------------------------
## --------------------------------------------------------------------------

### ----  Scenario 1 ----
phi_at_knots = 0.65-sqrt((Knots_data[,1]-3)^2/4 + (Knots_data[,2]-3)^2/3)/10
phi_vec = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[idx] = sum(weights*phi_at_knots)
}

library(plotly)
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec,nrow=length(station_tmp)), colors='#196cb5', opacity = 0.93, showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(phi_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03d5(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
# setwd('~/Desktop/')
orca(fig, file='./PyCode/Simulation_figures/phi_scenario1.png')

library(magick)
img <- image_read('~/Desktop/PyCode/Simulation_figures/phi_scenario1.png')
img <- image_trim(img)
print(img)
image_write(img, path='~/Desktop/PyCode/Simulation_figures/phi_scenario1_trimmed.png', format='png')

### ----  Scenario 2 ----
phi_at_knots = 0.65-sqrt((Knots_data[,1]-5.1)^2/5 + (Knots_data[,2]-5.3)^2/4)/11.6
phi_vec = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[idx] = sum(weights*phi_at_knots)
}

library(plotly)
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec,nrow=length(station_tmp)), colors='#196cb5', opacity = 0.93, showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(phi_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03d5(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
# setwd('~/Desktop/')
orca(fig, file='./PyCode/Simulation_figures/phi_scenario2.png')

library(magick)
img <- image_read('~/Desktop/PyCode/Simulation_figures/phi_scenario2.png')
img <- image_trim(img)
print(img)
image_write(img, path='~/Desktop/PyCode/Simulation_figures/phi_scenario2_trimmed.png', format='png')


### ----  Scenario 3 ----
bw=3
phi_at_knots = rep(NA,9)
for(i in 1:9){
  phi_at_knots[i] = 10*(0.5*mvtnorm::dmvnorm(Knots_data[i,], mean=c(2.5,3), sigma = matrix(2*c(1,0.2,0.2,1),ncol=2))+
                       0.5*mvtnorm::dmvnorm(Knots_data[i,], mean=c(7,7.5), sigma = matrix(2*c(1,-0.2,-0.2,1),ncol=2)))+0.37
}
phi_vec = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[idx] = sum(weights*phi_at_knots)
}

library(plotly)
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec,nrow=length(station_tmp)), colors='#196cb5', opacity = 0.9, showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(phi_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03d5(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
# setwd('~/Desktop/')
orca(fig, file='./PyCode/Simulation_figures/phi_scenario3.png')

library(magick)
img <- image_read('~/Desktop/PyCode/Simulation_figures/phi_scenario3.png')
img <- image_trim(img)
print(img)
image_write(img, path='~/Desktop/PyCode/Simulation_figures/phi_scenario3_trimmed.png', format='png')




## --------------------------------------------------------------------------
## ------------------------------ CI surfaces -------------------------------
## --------------------------------------------------------------------------

### ----  phi ----
bw = 4 # bandwidth
n_s=25*25
station_tmp = seq(0,10,length.out = 25)
Stations = expand.grid(x=station_tmp,y=station_tmp)

phi_at_knots = 0.65-sqrt((Knots_data[,1]-3)^2/4 + (Knots_data[,2]-3)^2/3)/10
phi_at_knots_Upper = c(0.31945,0.46446,0.42724,0.41897,0.49894,0.44551,0.44219,0.51052,0.51936)
phi_at_knots_Upper = rev(phi_at_knots_Upper)
phi_at_knots_Lower = c(0.46779,0.44175,0.36952,0.39709,0.44873,0.38729,0.34548,0.41486,0.25561)
phi_vec = rep(NA, n_s)
phi_vec_Upper = rep(NA, n_s)
phi_vec_Lower = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[idx] = sum(weights*phi_at_knots)
  phi_vec_Upper[idx] = sum(weights*phi_at_knots_Upper)
  phi_vec_Lower[idx] = sum(weights*phi_at_knots_Lower)
}

library(plotly)
color2 <- rep('#ed6755', length(phi_vec_Upper))
dim(color2) <- c(length(station_tmp),length(station_tmp))
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec,nrow=length(station_tmp)), colors='#196cb5', opacity = 0.93, showscale=FALSE) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec_Upper,nrow=length(station_tmp)), opacity = 0.73,
              cmin = min(phi_vec_Upper), cmax = max(phi_vec_Upper), colorscale = list(c(0,1),c("rgb(255,112,184)","rgb(128,0,64)")), showscale=FALSE) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(phi_vec_Lower,nrow=length(station_tmp)), opacity = 0.73,
              cmin = min(phi_vec_Lower), cmax = max(phi_vec_Lower), colorscale = list(c(0,1),c("rgb(255,112,184)","rgb(128,0,64)")), showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(phi_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03d5(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig



### ----  rho ----
bw = 4 # bandwidth
n_s=25*25
station_tmp = seq(0,10,length.out = 25)
Stations = expand.grid(x=station_tmp,y=station_tmp)

rho_at_knots = (-1 + 0.3*Knots_data[,1] + 0.4*Knots_data[,2])/4
rho_at_knots_Upper = c(0.34934,0.49909,0.64739,0.52146,0.62689,0.77059,0.65444,0.84005,0.92266)
rho_at_knots_Lower = c(0.29361,0.42845,0.55548,0.45554,0.56419,0.69956,0.56339,0.70543,0.80588)
rho_vec = rep(NA, n_s)
rho_vec_Upper = rep(NA, n_s)
rho_vec_Lower = rep(NA, n_s)
for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  rho_vec[idx] = sum(weights*rho_at_knots)
  rho_vec_Upper[idx] = sum(weights*rho_at_knots_Upper)
  rho_vec_Lower[idx] = sum(weights*rho_at_knots_Lower)
}

library(plotly)
color2 <- rep('#ed6755', length(rho_vec_Upper))
dim(color2) <- c(length(station_tmp),length(station_tmp))
fig <- plot_ly(width=700, height=500) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(rho_vec,nrow=length(station_tmp)), colors='#196cb5', opacity = 0.93, showscale=FALSE) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(rho_vec_Upper,nrow=length(station_tmp)), opacity = 0.73,
              cmin = min(rho_vec_Upper), cmax = max(rho_vec_Upper), colorscale = list(c(0,1),c("rgb(255,112,184)","rgb(128,0,64)")), showscale=FALSE) %>% 
  add_surface(x = station_tmp, y = station_tmp, z = matrix(rho_vec_Lower,nrow=length(station_tmp)), opacity = 0.73,
              cmin = min(rho_vec_Lower), cmax = max(rho_vec_Lower), colorscale = list(c(0,1),c("rgb(255,112,184)","rgb(128,0,64)")), showscale=FALSE) %>% 
  # add_surface(x=station_tmp,y=station_tmp,z=matrix(0,nrow=length(station_tmp),ncol=length(station_tmp)),surfacecolor=matrix(rho_vec,nrow=length(station_tmp)), colorscale='Portland') %>%
  layout(
    scene = list(xaxis = list(title = "s<sub>1</sub>"), yaxis = list(title = "s<sub>2</sub>"), 
                 zaxis = list(title = "\u03C1(<b>s</b>)"),
                 camera = list(eye = list(x = 1.7, y = -1.7, z = 0.7)))
  )#%>% config(mathjax = 'cdn')
fig
