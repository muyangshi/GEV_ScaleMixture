## ** Use importance sampling **

weights_fun<-function(d,radius,h=1, cutoff=TRUE){
  tmp = exp(-d**2/(2*h))
  if(cutoff) tmp[d>radius] = 0
  
  return(tmp/sum(tmp))
}

circleFun <- function(center = c(0,0),diameter = 1, npoints = 200){
  r = diameter / 2
  tt <- seq(0,2*pi,length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

setwd('~/Desktop/')
loc_tmp=seq(0,10,length.out = 7)[c(2,4,6)]
Knots_data0 = expand.grid(x=loc_tmp, y=loc_tmp)

# ****  Isometric grid ****
xmin <- 0
xmax <- 10
ymin <- xmin
ymax <- xmax
Ngrid <- 3^2
x_vals <- seq(from = xmin + 1, to = xmax + 1, length = 2*sqrt(Ngrid))
y_vals <- seq(from = ymin + 1, to = ymax + 1, length = 2*sqrt(Ngrid))
isometric_grid <- rbind(
  expand.grid(x = x_vals[seq(from=1,to=length(x_vals),by=2)],
              y = y_vals[seq(from=1,to=length(y_vals),by=2)]),
  expand.grid(x = x_vals[seq(from=2,to=length(x_vals),by=2)],
              y = y_vals[seq(from=2,to=length(y_vals),by=2)])
)
isometric_grid <- isometric_grid[isometric_grid$x < xmax & isometric_grid$y < ymax,]
# plot(isometric_grid, asp = 1, pch = "+", cex = 1.5, xlim = c(0,10), ylim = c(0,10))
# abline(v = c(0,10), h = c(0,10), col = "gray")
# for(g in 1:nrow(isometric_grid)) plotrix::draw.circle(x = isometric_grid[g,1], y = isometric_grid[g,2], radius = 2.5, border = 2)
Knots_data = isometric_grid


## --------------------------------------------------------------------------
## ----------------------------- phi surface ------------------------------
## --------------------------------------------------------------------------

# phi_at_knots = c(0.39, 0.49, 0.53, 0.501, 0.477, 0.516, 0.557, 0.561, 0.58)
phi_at_knots = c(0.4148, 0.481, 0.5232, 0.4921, 0.4989, 0.5245, 0.5499, 0.5539, 0.5692, 0.4612, 0.5069, 0.528, 0.5398)

bw = 4 # bandwidth
resolution = 50
n_s=resolution*resolution
station_tmp = seq(0,10,length.out = resolution)
Stations = expand.grid(x=station_tmp,y=station_tmp)
phi_vec = rep(NA, n_s)

for(idx in 1:n_s) {
  d_tmp = as.matrix(dist(rbind(Stations[idx,],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[idx] = sum(weights*phi_at_knots)
}

set.seed(123)
sample_points <- data.frame(x=Knots_data0$x + rnorm(9,sd=0.5), y=Knots_data0$y + rnorm(9,sd=0.5), pname = paste0('Point ', 1:9))
plot_data <- data.frame(x=Stations$x, y=Stations$y, phi_vec=phi_vec)
library(ggplot2)
library(ggh4x)
brks = c(seq(0.42,0.5,length.out=6),round(seq(0.5,0.57,length.out=6)[-1],3))
g = ggplot(plot_data, aes(x,y)) +
  # geom_raster(aes(fill = phi_vec)) +
  geom_contour_filled(aes(z = phi_vec), breaks=brks)+
  # scale_fill_gradientn(colours = brewer.pal(11,"RdBu")) +
  geom_point(data = Knots_data, aes(x,y), shape=3, colour="#c9a800", size=3, stroke = 1) +
  geom_point(data = sample_points, aes(x,y, shape=pname), colour="black", size=2) +
  scale_fill_brewer(palette = "RdBu", direction = -1, name=expression(phi(s))) +
  scale_shape_manual(values=c(20, 17, 15, 7, 8, 4, 16, 17,14), name="Sample points") + 
  geom_contour(aes(z=phi_vec), breaks=0.5, linetype='dashed', colour='black') +
  coord_cartesian(xlim = c(0,10), ylim = c(0,10)) +
  theme(legend.position = "right", panel.background = element_rect(colour = "black", fill=NA),
        plot.title = element_text(hjust = 0.5, size=14))+
  guides(shape = guide_legend(order = 2, ncol=2),fill = guide_legend(order = 1, ncol=2))+
  force_panelsizes(rows = unit(3.75, "in"),
                   cols = unit(3.75, "in"))

radius = rep(2.5, nrow(Knots_data))
# radius[1] = 4; radius[13] = 4 # lower left and (7,7)
circle_dat <- cbind(circleFun(as.numeric(Knots_data[1,1:2]),radius[1]*2), ind=1)
for (i in 2:nrow(Knots_data)){
  circle_dat <- rbind(circle_dat, cbind(circleFun(as.numeric(Knots_data[i,1:2]),radius[i]*2), ind=i))
}

g+geom_path(data=circle_dat, aes(x=x,y=y,group=ind), colour='gray',alpha=0.8) +
  labs(title='Scenario 1') #+ geom_path(data=circle_dat[circle_dat$ind==1 | circle_dat$ind==13,], aes(x=x,y=y,group=ind), colour='gray40',alpha=0.9, size=1) 
ggsave(
  filename = paste0("~/Desktop/Nonstat/Sec4_figures/phi1.pdf"), 
  width = 7, height = 4.5
)

## --------------------------------------------------------------------------
## ---------------------------- Wendland basis ------------------------------
## --------------------------------------------------------------------------
## fields::Wendland(d, theta = 1, dimension, k, derivative=0, phi=NA)
## theta: the range where the basis value is non-zero, i.e. [0, theta]
## dimension: dimension of locations
## k: smoothness of the function at zero.
wendland_weights_fun<-function(d,radius,k=2){
  if(k>0) tmp = fields::Wendland(d, theta = radius, dimension=2, k=k)
  if(k==0) tmp = ifelse(1>d/radius, (1-d/radius)^2,0)
  return(tmp/sum(tmp))
}

bump_weights_fun <- function(d, radius, beta=0.2){
  ampl=1
  tmp = rep(NA, length(d))
  tmp[d>radius] = 0
  tmp[d<=radius] = ampl*exp(beta - beta/(1-d[d<=radius]^2/radius^2))
  
  return(tmp/sum(tmp))
}

## ---------------------------------
## ---------- Generate Sk ----------
## ---------------------------------
rlevy<-function (n = 1, m = 0, s = 1) 
{
  if (any(s < 0)) 
    stop("s must be positive")
  s/qnorm(runif(n)/2)^2 + m
}

set.seed(2)
loc <- 0; gamma <- 0.3
S <- rlevy(nrow(Knots_data), m=loc, s=gamma)

# ----------------------------------
## ------ Average using basis ------
## ---------------------------------
x <- seq(0,10,length.out = 80); y <- seq(0,10,length.out = 80)
sites <- expand.grid(x=x,y=y)

radius = 4
R.s <- rep(NA, nrow(sites))
for(iter in 1:nrow(sites)){
  h_tmp <- fields::rdist(sites[iter,],Knots_data)
  R.s[iter]<- sum(wendland_weights_fun(h_tmp[1,], radius, k=0)*S)
  # R.s[iter]<- sum(bump_weights_fun(h_tmp[1,], radius, beta=0.5)*S)
}

plot_gg <- ggplot(sites) + geom_raster(aes(x,y,fill=R.s)) + 
  scale_fill_gradientn(colours = terrain.colors(12)) +
  scale_y_continuous(expand = c(0,0)) + scale_x_continuous(expand = c(0,0))+
  labs(fill=expression(R(s)), title = paste0('At knots: ',paste0(round(S,2), collapse=', '),'\n', 'radius: ', radius) ) +
  theme(legend.position="bottom",
        plot.title = element_text(hjust = 0.5, size=14),
        panel.background = element_rect(colour = "black", fill=NA))+
  force_panelsizes(rows = unit(3.75, "in"),
                   cols = unit(3.75, "in"))

circle_dat <- cbind(circleFun(as.numeric(Knots_data[1,1:2]),radius*2), ind=1)
for (i in 2:nrow(Knots_data)){
  circle_dat <- rbind(circle_dat, cbind(circleFun(as.numeric(Knots_data[i,1:2]),radius*2), ind=i))
}

plot_gg+geom_path(data=circle_dat, aes(x=x,y=y,group=ind), colour='gray',alpha=0.8) +
  geom_point(data = Knots_data, aes(x,y), shape=3, colour="#bf2102", size=3, stroke = 1)
ggsave(
  filename = paste0("~/Desktop/radius2.png"), 
  width = 5.5, height = 5.5
)






## --------------------------------------------------------------------------
## ------------------------------ Same radius -------------------------------
## --------------------------------------------------------------------------
## Global parameters
phi_vec <- rep(NA,nrow(sample_points))
for (i in 1:nrow(sample_points)){
  d_tmp = as.matrix(dist(rbind(sample_points[i,1:2],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[i] = sum(weights*phi_at_knots)
}

## K=9 knots
k=0; radius = 2.5
Weight_matrix <- matrix(NA, nrow=nrow(sample_points), ncol = nrow(Knots_data))
for( i in 1:nrow(sample_points)){
  h_tmp <- fields::rdist(sample_points[i,1:2],Knots_data)
  Weight_matrix[i,] <- wendland_weights_fun(h_tmp[1,], radius, k=k)
}



## Generate empirical CDF
### Conclusion: Sample size = 20k gives good estimate of thresh_98 but not the cdf after thresh_98.
gamma_vec <- apply(Weight_matrix, 1, function(x) (sum((x*gamma)^{1/2}))^2) 
rlevy<-function (n = 1, m = 0, s = 1) 
{
  if (any(s < 0)) 
    stop("s must be positive")
  s/qnorm(runif(n)/2)^2 + m
}

## Generate N pairs (R_1^phi_1*W_1, R_2^phi_2*W_2)
N =10000000
Finverse<-function(u) {return(1/(1-u))}
Realizations<-matrix(NA,nrow=N,ncol=9)
W_Realizations<-matrix(NA,nrow=N,ncol=9)
cat('When phi9=',phi9,', phi8=',phi8,', and phi6=',phi6,':\n')
Cov <- matrix(c(1.00000000e+00, 7.15373903e-02, 1.38142770e-03, 6.32405788e-02,
                1.91119534e-02, 8.84934016e-04, 1.55628332e-03, 1.89497154e-03,
                7.47057517e-04,
                7.15373903e-02, 1.00000000e+00, 5.55528456e-02, 4.70440971e-02,
                1.97959441e-01, 3.40071073e-02, 2.91773935e-03, 1.01826326e-02,
                1.58184964e-02,
                1.38142770e-03, 5.55528456e-02, 1.00000000e+00, 1.56674616e-03,
                2.93741878e-02, 2.11167982e-01, 2.91637451e-04, 2.25819733e-03,
                2.48925073e-02,
                6.32405788e-02, 4.70440971e-02, 1.56674616e-03, 1.00000000e+00,
                1.26058085e-01, 3.59732164e-03, 7.54302975e-02, 6.30846946e-02,
                9.05145278e-03,
                1.91119534e-02, 1.97959441e-01, 2.93741878e-02, 1.26058085e-01,
                1.00000000e+00, 7.91202547e-02, 3.02165017e-02, 1.24079002e-01,
                1.18461799e-01,
                8.84934016e-04, 3.40071073e-02, 2.11167982e-01, 3.59732164e-03,
                7.91202547e-02, 1.00000000e+00, 1.87469198e-03, 1.64675728e-02,
                2.33305163e-01,
                1.55628332e-03, 2.91773935e-03, 2.91637451e-04, 7.54302975e-02,
                3.02165017e-02, 1.87469198e-03, 1.00000000e+00, 2.80991591e-01,
                1.44876870e-02,
                1.89497154e-03, 1.01826326e-02, 2.25819733e-03, 6.30846946e-02,
                1.24079002e-01, 1.64675728e-02, 2.80991591e-01, 1.00000000e+00,
                1.18535825e-01,
                7.47057517e-04, 1.58184964e-02, 2.48925073e-02, 9.05145278e-03,
                1.18461799e-01, 2.33305163e-01, 1.44876870e-02, 1.18535825e-01,
                1.00000000e+00
                ),ncol=9)
for(j in 1:N){
  if(j %% 10000==0) cat('--',j,'\n')
  Z<-mvtnorm::rmvnorm(1, sigma = Cov)
  W_vec<-Finverse(pnorm(Z))
  S <- rlevy(nrow(Knots_data), m=0, s=gamma)
  R_vec <- apply(Weight_matrix,1, function(x) sum(x*S))
  
  X<-R_vec^phi_vec*W_vec
  Realizations[j,]<-X
  W_Realizations[j,] <- W_vec
}
save(Realizations, W_Realizations, file="./Realizations_equal.RData")
## Theoretical bounds
# col1 = W_Realizations[,1]^{1/(2*phi_1)}
# col2 = W_Realizations[,2]^{1/(2*phi_2)}
# tmp = cbind(col1*(1-1/(2*phi_1)),col2*(1-1/(2*phi_2)))
# theo_chi1= mean(apply(tmp,1, min)) 
# theo_chi2= mean(apply(tmp,1, max)) 
# v1 <- (w1*gamma)^{1/2}/sum((w1*gamma)^{1/2})
# v2 <- (w2*gamma)^{1/2}/sum((w2*gamma)^{1/2})
# lower_bound <- theo_chi1*sum(apply(cbind(v1,v2),1,min))
# upper_bound <- theo_chi2*sum(apply(cbind(v1,v2),1,max))
# 

load('~/Desktop/Nonstat/Sec4_figures/Realizations_equal.RData')
## Numerical limits
Ecdf_tmp <- apply(Realizations, 2, ecdf)
# Ecdf_tmp9 <-ecdf(Realizations[,1])
# Ecdf_tmp8 <-ecdf(Realizations[,2])
# Ecdf_tmp6 <-ecdf(Realizations[,3])

Margin_Unif = matrix(NA, nrow(Realizations), ncol(Realizations))
for(i in 1:ncol(Realizations)){
  Margin_Unif[,i] = Ecdf_tmp[[i]](Realizations[,i])
}
# Margin_Unif = cbind(Ecdf_tmp9(Realizations[,1]), Ecdf_tmp8(Realizations[,2]),
#                     Ecdf_tmp6(Realizations[,3]))  # Marginal Uniform Scale
u = c(seq(0.9,0.9937895,length.out = 30),seq(0.995,0.99999,length.out = 30))
# u = c(seq(0.9,0.999,length.out = 20),seq(0.99901,0.99999,length.out = 70))
Chi = matrix(NA,nrow=length(u), ncol=6)
for(i in 1:length(u)){
  if(i %% 10==0) cat('--',i,'\n')
  Chi[i,1]=mean(apply(Margin_Unif,1,function(x) x[9]>u[i]&x[6]>u[i]))/(1-u[i]) #AI
  Chi[i,2]=mean(apply(Margin_Unif,1,function(x) x[9]>u[i]&x[5]>u[i]))/(1-u[i]) #AI weaker than previous
  Chi[i,3]=mean(apply(Margin_Unif,1,function(x) x[7]>u[i]&x[8]>u[i]))/(1-u[i]) #AD weak
  Chi[i,4]=mean(apply(Margin_Unif,1,function(x) x[1]>u[i]&x[2]>u[i]))/(1-u[i]) #AI
  Chi[i,5]=mean(apply(Margin_Unif,1,function(x) x[2]>u[i]&x[3]>u[i]))/(1-u[i]) #AI stronger or weaker than previous
  Chi[i,6]=mean(apply(Margin_Unif,1,function(x) x[1]>u[i]&x[4]>u[i]))/(1-u[i]) #AI 
}
save(Chi, file='~/Desktop/Nonstat/Sec4_figures/Equal_radius1.RData')

## ********************************
## ********** VISUALIZE ***********
## ********************************
load('~/Desktop/Nonstat/Sec4_figures/Realizations_equal.RData')
load('~/Desktop/Nonstat/Sec4_figures/Equal_radius1.RData')
# u = c(seq(0.9,0.9937895,length.out = 39),seq(0.995,0.99999,length.out = 50))
u = c(seq(0.9,0.9937895,length.out = 30),seq(0.995,0.99999,length.out = 30))
plot(u, Chi[,1], type='l', col=1, lwd=2, ylim=c(0,0.2))
points(u, Chi[,2], type='l', col=2, lwd=2)
points(u, Chi[,3], type='l', col=3, lwd=2)
# *In the same radius, give you higer dependence even if phi<0.5*
points(u, Chi[,4], type='l', col=4, lwd=2)
points(u, Chi[,5], type='l', col=5, lwd=2) 
points(u, Chi[,6], type='l', col=6, lwd=2) 

# ----------------------
# ---- Point 6 vs 9 ----
# ----------------------
ind = 1
smoothingSpline = smooth.spline(u[40:length(u)], Chi[40:length(u),ind], spar=0.8)
plot(smoothingSpline,lwd=2)
lines(smoothingSpline,lwd=2, col='blue')
points(u[40:length(u)], Chi[40:length(u),ind], pch=20, col='red')

chi_u_sequence <- data.frame(u=u,chi=c(Chi[1:39,ind], smoothingSpline$y))
limit_chi <-  data.frame(u=1,chi=0)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,6])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_96.pdf', width=3,height=3)
print(plot_gg)
dev.off()

## eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
alpha <- 1/2
eta_W <- (1+Cov[9,6])/2
phi_1 <- phi_vec[9]; phi_2 <- phi_vec[6]
# phi_1/(alpha*eta_W) < 2
lower_eta <- alpha*eta_W/phi_1; upper_eta <- alpha/phi_2

Ecdf_tmp1 <-ecdf(Realizations[,9])
Ecdf_tmp2 <-ecdf(Realizations[,6])
Margin_Unif = cbind(Ecdf_tmp1(Realizations[,9]), Ecdf_tmp2(Realizations[,6]))  # Marginal Uniform Scale

q<-seq(0.00008,0.1,length.out = 400)
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q,1/eta_inverse,type='l',xlim=c(0.99,1),ylim=c(0,1),
     xlab='1-p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[9,6])))
abline(h=lower_eta, lty=2, col='red', lwd=2)
abline(h=upper_eta, lty=2, col='red', lwd=2)
grid()

smoothingSpline = smooth.spline(1-q[1:6], 1/eta_inverse[1:6], spar=0.41)
plot(smoothingSpline,lwd=2, ylim=c(0.532,0.597))
lines(smoothingSpline,lwd=2, col='orange')
points(1-q[1:38], 1/eta_inverse[1:38], pch=20, col='red')

smoothingSpline1 = smooth.spline(1-q[7:18], 1/eta_inverse[7:18], spar=0.41)
plot(smoothingSpline1,lwd=2, ylim=c(0.532,0.597))
lines(smoothingSpline1,lwd=2, col='blue')
points(1-q[7:18], 1/eta_inverse[7:18], pch=20, col='red')

smoothingSpline2 = smooth.spline(1-q[19:length(q)], 1/eta_inverse[19:length(q)], spar=0.85)
plot(smoothingSpline2,lwd=2, ylim=c(0.532,0.597))
lines(smoothingSpline2,lwd=2, col='green')
points(1-q[19:length(q)], 1/eta_inverse[19:length(q)], pch=20, col='red')

eta_u_sequence <- data.frame(u=c(rev(1-q)[-(383:395)],1),
                             eta=c(smoothingSpline2$y,smoothingSpline$y[-1],1/eta_inverse[1]))
save(eta_inverse, eta_u_sequence, file='~/Desktop/Nonstat/Sec4_figures/eta_96.RData')

limit_eta <- data.frame(u=1, eta=1/eta_inverse[1])
plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept=lower_eta,linetype='dashed', colour='red',size=1.2)+
  geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_eta),linetype='dashed', colour='red',size=1.2)+
  geom_point(data = limit_eta, mapping = aes(x=u, y=eta), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,6])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.25,1.02),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_96_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()


# ----------------------
# ---- Point 9 vs 5 ----
# ----------------------
ind = 2
smoothingSpline = smooth.spline(u[40:length(u)], Chi[40:length(u),ind], spar=0.8)
plot(smoothingSpline,lwd=2)
lines(smoothingSpline,lwd=2, col='blue')
points(u[40:length(u)], Chi[40:length(u),ind],pch=20,col='red')

chi_u_sequence <- data.frame(u=u,chi=c(Chi[1:39,ind], smoothingSpline$y))
limit_chi <-  data.frame(u=1,chi=0)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[5],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[5], Cov[9,5])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_95.pdf', width=3,height=3)
print(plot_gg)
dev.off()

## eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
alpha <- 1/2
eta_W <- (1+Cov[9,5])/2
phi_1 <- phi_vec[9]; phi_2 <- phi_vec[5] ## no shared non-zero weights
# phi_1/(alpha*eta_W) < 2
rho=Cov[9,5]
lower_eta <- alpha*eta_W/phi_1; upper_eta <- alpha/phi_2

N =5e+07

Realizations_95<-matrix(NA,nrow=N,ncol=2)
W_Realizations_95<-matrix(NA,nrow=N,ncol=2)
cat('When phi1=',phi_1,', phi2=',phi_2,', and rho=',rho,':\n')
weights_matrix <- Weight_matrix[c(9,5),]
cov <- matrix(c(1,rho,rho,1),ncol=2)
for(j in 1:N){
  if(j %% 1000000==0) cat('--',j,'\n')
  Z<-mvtnorm::rmvnorm(1, sigma = cov)
  W_vec<-Finverse(pnorm(Z))
  S <- rlevy(nrow(Knots_data), m=0, s=gamma)
  R_vec <- apply(weights_matrix,1, function(x) sum(x*S))
  
  X<-R_vec^c(phi_1,phi_2)*W_vec
  Realizations_95[j,]<-X
  W_Realizations_95[j,] <- W_vec
}

Ecdf_tmp1 <-ecdf(Realizations_95[,1])
Ecdf_tmp2 <-ecdf(Realizations_95[,2])
Margin_Unif = cbind(Ecdf_tmp1(Realizations_95[,1]), 
                    Ecdf_tmp2(Realizations_95[,2]))  # Marginal Uniform Scale

# Ecdf_tmp1 <-ecdf(Realizations[,9])
# Ecdf_tmp2 <-ecdf(Realizations[,5])
# Margin_Unif = cbind(Ecdf_tmp1(Realizations[,9]), Ecdf_tmp2(Realizations[,5]))  # Marginal Uniform Scale

q<-seq(0.00004,0.1,length.out = 400)
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q[-1],1/eta_inverse[-1],type='l',xlim=c(0.9,1),ylim=c(0.5,0.6),
     xlab='1-p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[9,6])))
abline(h=lower_eta, lty=2, col='red', lwd=2)
abline(h=upper_eta, lty=2, col='red', lwd=2)
grid()

smoothingSpline = smooth.spline(1-q[c(3,6:140)], 1/eta_inverse[c(3,6:140)], spar=0.65)
plot(smoothingSpline,lwd=2, ylim=c(0.50,0.5279639))
lines(smoothingSpline,lwd=2, col='blue')
points(1-q[c(2,6:140)], 1/eta_inverse[c(2,6:140)], col='green', pch=20)

smoothingSpline2 = smooth.spline(1-q[140:length(q)], 1/eta_inverse[140:length(q)], spar=0.85)
plot(smoothingSpline2,lwd=2, ylim=c(0.5165, 0.5279639), xlim=c(0.9,1))
lines(smoothingSpline2,lwd=2, col='blue')
points(1-q, 1/eta_inverse, col='red', pch=20)

eta_u_sequence <- data.frame(u=c(smoothingSpline2$x, smoothingSpline$x, 1),eta=c(smoothingSpline2$y, smoothingSpline$y, 0.5250466))
eta_u_sequence <- eta_u_sequence[c(1:344, 398),]
save(eta_inverse, eta_u_sequence, file='~/Desktop/Nonstat/Sec4_figures/eta_95.RData')

plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept=lower_eta,linetype='dashed', colour='red',size=1.2)+
  geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_eta),linetype='dashed', colour='red',size=1.2)+
  # geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,5])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.5,0.6),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))

pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_95_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()






# ----------------------
# ---- Point 2 vs 3 ----
# ----------------------
load("/Users/LikunZhang/Desktop/Nonstat/Sec4_figures/Equal_radius.RData")
ind = 5
u = c(seq(0.9,0.9937895,length.out = 39),seq(0.995,0.99999,length.out = 50))
smoothingSpline = smooth.spline(c(u[35:length(u)],1), c(Chi[35:length(u),ind],0), spar=0.9)
plot(smoothingSpline,lwd=2,ylim=c(0,0.03))
lines(smoothingSpline,lwd=2, col='blue')
points(u[35:length(u)], Chi[35:length(u),ind], pch=20, col='red')

chi_u_sequence <- data.frame(u=c(u,1),
                    chi=c(Chi[1:34,ind],smoothingSpline$y[-length(smoothingSpline$y)], 0))
limit_chi <-  data.frame(u=1,chi=0)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[2],'=',%.3f,', ',phi[3],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[2],phi_vec[3], Cov[2,3])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))

pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_23.pdf', width=3,height=3)
print(plot_gg)
dev.off()

## eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
load('~/Desktop/Nonstat/Sec4_figures/Realizations_equal.RData')
load('~/Desktop/Nonstat/Sec4_figures/Equal_radius1.RData')
u = c(seq(0.9,0.9937895,length.out = 30),seq(0.995,0.99999,length.out = 30))
alpha <- 1/2
eta_W <- (1+Cov[2,3])/2
phi_1 <- phi_vec[2]; phi_2 <- phi_vec[3]
# eta_W < mean(phi_1/alpha, phi_2/alpha)
upper_eta <- 1/((1-phi_1/alpha)/(2*eta_W)+1); lower_eta <- 1/(2-phi_1/alpha)

Ecdf_tmp1 <-ecdf(Realizations[,2])
Ecdf_tmp2 <-ecdf(Realizations[,3])
Margin_Unif = cbind(Ecdf_tmp1(Realizations[,2]), Ecdf_tmp2(Realizations[,3]))  # Marginal Uniform Scale

q<-c(5.7e-7, seq(0.000001,0.1,length.out = 400))
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q,1/eta_inverse,type='l',xlim=c(0.9,1),ylim=c(0,1),
     xlab='p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[2,3])))
abline(h=lower_eta, lty=2, col='red', lwd=2)
abline(h=upper_eta, lty=2, col='blue', lwd=2)
grid()

eta_u_sequence <- data.frame(u=c(rev(1-q),1),
                             eta=c(rev(1/eta_inverse), lower_eta))
save(eta_inverse, eta_u_sequence, file='~/Desktop/Nonstat/Sec4_figures/eta_23.RData')

limit_eta <- data.frame(u=1, eta=lower_eta)
plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept=lower_eta,linetype='dashed', colour='red',size=1.2)+
  geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_eta),linetype='dashed', colour='red',size=1.2)+
  geom_point(data = limit_eta, mapping = aes(x=u, y=eta), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[2],'=',%.3f,', ',phi[3],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[2],phi_vec[3], Cov[2,3])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.25,1.02),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_23_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()


# ----------------------
# ---- Point 1 vs 2 ----
# ----------------------
load('~/Desktop/Nonstat/Sec4_figures/Realizations_equal.RData')
load('~/Desktop/Nonstat/Sec4_figures/Equal_radius1.RData')
u = c(seq(0.9,0.9937895,length.out = 30),seq(0.995,0.99999,length.out = 30))

ind = 4
smoothingSpline = smooth.spline(u[40:length(u)], c(Chi[40:(length(u)-1),ind],0.001), spar=0.8)
plot(smoothingSpline,lwd=2,ylim=c(0,0.035),xlim=c(0.9965,1))
lines(smoothingSpline,lwd=2, col='blue')
points(u[40:length(u)], Chi[40:length(u),ind], pch=20, col='red')

chi_u_sequence <- data.frame(u=c(u,1),chi=c(Chi[1:39,ind], smoothingSpline$y,0))
limit_chi <-  data.frame(u=1,chi=0)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[2],'=',%.3f,', ',phi[1],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[2],phi_vec[1], Cov[1,2])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_12.pdf', width=3,height=3)
print(plot_gg)
dev.off()

## eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
alpha <- 1/2
eta_W <- (1+Cov[1,2])/2
rho <- Cov[1,2]
phi_1 <- phi_vec[1]; phi_2 <- phi_vec[2]
# eta_W < phi_1/alpha
lower_eta <- phi_1/alpha; upper_eta <- phi_2/alpha

N =3e+08

Realizations_12<-matrix(NA,nrow=N,ncol=2)
# W_Realizations_12<-matrix(NA,nrow=N,ncol=2)
cat('When phi1=',phi_1,', phi2=',phi_2,', and rho=',rho,':\n')
weights_matrix <- Weight_matrix[c(1,2),]
cov <- matrix(c(1,rho,rho,1),ncol=2)
for(j in 1:N){
  if(j %% 10000000==0) cat('--',j,'\n')
  Z<-mvtnorm::rmvnorm(1, sigma = cov)
  W_vec<-Finverse(pnorm(Z))
  S <- rlevy(nrow(Knots_data), m=0, s=gamma)
  R_vec <- apply(weights_matrix,1, function(x) sum(x*S))

  X<-R_vec^c(phi_1,phi_2)*W_vec
  Realizations_12[j,]<-X
  # W_Realizations_12[j,] <- W_vec
}

Ecdf_tmp1 <-ecdf(Realizations_12[,1])
Ecdf_tmp2 <-ecdf(Realizations_12[,2])
Margin_Unif = cbind(Ecdf_tmp1(Realizations_12[,1]),
                    Ecdf_tmp2(Realizations_12[,2]))  # Marginal Uniform Scale


Ecdf_tmp1 <-ecdf(Realizations[,1])
Ecdf_tmp2 <-ecdf(Realizations[,2])
Margin_Unif = cbind(Ecdf_tmp1(Realizations[,1]), Ecdf_tmp2(Realizations[,2]))  # Marginal Uniform Scale

q<-c(1.6e-6, seq(0.000001,0.1,length.out = 400))
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q,1/eta_inverse,type='l',xlim=c(0.9,1),ylim=c(0,1),
     xlab='1-p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[1,2])))
abline(h=lower_eta, lty=2, col='red', lwd=2)
abline(h=upper_eta, lty=2, col='red', lwd=2)
grid()


eta_u_sequence <- data.frame(u=c(rev(1-q[-2])),
                             eta=c(rev(1/eta_inverse[-2],)))
save(eta_inverse, eta_u_sequence, file='~/Desktop/Nonstat/Sec4_figures/eta_12.RData')

limit_eta <- data.frame(u=1, eta=tail(eta_u_sequence$eta,1))
plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept=lower_eta,linetype='dashed', colour='red',size=1.2)+
  geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_eta),linetype='dashed', colour='red',size=1.2)+
  geom_point(data = limit_eta, mapping = aes(x=u, y=eta), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,6])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.25,1.02),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_96_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()







## --------------------------------------------------------------------------
## ---------------------------- Different radii -----------------------------
## --------------------------------------------------------------------------
## Global parameters
phi_vec <- rep(NA,nrow(sample_points))
for (i in 1:nrow(sample_points)){
  d_tmp = as.matrix(dist(rbind(sample_points[i,1:2],Knots_data),upper=TRUE))[-1,1]
  weights = weights_fun(d_tmp,radius,bw,cutoff=FALSE)
  phi_vec[i] = sum(weights*phi_at_knots)
}

## K=9 knots
k=0; radius = rep(2.5, nrow(Knots_data))
radius[1] = 4; radius[13] = 4 # lower left and (7,7)
Weight_matrix <- matrix(NA, nrow=nrow(sample_points), ncol = nrow(Knots_data))
for( i in 1:nrow(sample_points)){
  h_tmp <- fields::rdist(sample_points[i,1:2],Knots_data)
  Weight_matrix[i,] <- wendland_weights_fun(h_tmp[1,], radius, k=k)
}



## Generate empirical CDF
### Conclusion: Sample size = 20k gives good estimate of thresh_98 but not the cdf after thresh_98.
gamma_vec <- apply(Weight_matrix, 1, function(x) (sum((x*gamma)^{1/2}))^2) 
rlevy<-function (n = 1, m = 0, s = 1) 
{
  if (any(s < 0)) 
    stop("s must be positive")
  s/qnorm(runif(n)/2)^2 + m
}

## Generate N pairs (R_1^phi_1*W_1, R_2^phi_2*W_2)
N =10000000
Finverse<-function(u) {return(1/(1-u))}
Realizations<-matrix(NA,nrow=N,ncol=9)
W_Realizations<-matrix(NA,nrow=N,ncol=9)
cat('When phi9=',phi9,', phi8=',phi8,', and phi6=',phi6,':\n')
Cov <- matrix(c(1.00000000e+00, 7.15373903e-02, 1.38142770e-03, 6.32405788e-02,
                1.91119534e-02, 8.84934016e-04, 1.55628332e-03, 1.89497154e-03,
                7.47057517e-04,
                7.15373903e-02, 1.00000000e+00, 5.55528456e-02, 4.70440971e-02,
                1.97959441e-01, 3.40071073e-02, 2.91773935e-03, 1.01826326e-02,
                1.58184964e-02,
                1.38142770e-03, 5.55528456e-02, 1.00000000e+00, 1.56674616e-03,
                2.93741878e-02, 2.11167982e-01, 2.91637451e-04, 2.25819733e-03,
                2.48925073e-02,
                6.32405788e-02, 4.70440971e-02, 1.56674616e-03, 1.00000000e+00,
                1.26058085e-01, 3.59732164e-03, 7.54302975e-02, 6.30846946e-02,
                9.05145278e-03,
                1.91119534e-02, 1.97959441e-01, 2.93741878e-02, 1.26058085e-01,
                1.00000000e+00, 7.91202547e-02, 3.02165017e-02, 1.24079002e-01,
                1.18461799e-01,
                8.84934016e-04, 3.40071073e-02, 2.11167982e-01, 3.59732164e-03,
                7.91202547e-02, 1.00000000e+00, 1.87469198e-03, 1.64675728e-02,
                2.33305163e-01,
                1.55628332e-03, 2.91773935e-03, 2.91637451e-04, 7.54302975e-02,
                3.02165017e-02, 1.87469198e-03, 1.00000000e+00, 2.80991591e-01,
                1.44876870e-02,
                1.89497154e-03, 1.01826326e-02, 2.25819733e-03, 6.30846946e-02,
                1.24079002e-01, 1.64675728e-02, 2.80991591e-01, 1.00000000e+00,
                1.18535825e-01,
                7.47057517e-04, 1.58184964e-02, 2.48925073e-02, 9.05145278e-03,
                1.18461799e-01, 2.33305163e-01, 1.44876870e-02, 1.18535825e-01,
                1.00000000e+00
),ncol=9)
for(j in 1:N){
  if(j %% 10000==0) cat('--',j,'\n')
  Z<-mvtnorm::rmvnorm(1, sigma = Cov)
  W_vec<-Finverse(pnorm(Z))
  S <- rlevy(nrow(Knots_data), m=0, s=gamma)
  R_vec <- apply(Weight_matrix,1, function(x) sum(x*S))
  
  X<-R_vec^phi_vec*W_vec
  Realizations[j,]<-X
  W_Realizations[j,] <- W_vec
}
save(Realizations, W_Realizations, file="./Realizations_different.RData")

## Theoretical bounds
# col1 = W_Realizations[,1]^{1/(2*phi_1)}
# col2 = W_Realizations[,2]^{1/(2*phi_2)}
# tmp = cbind(col1*(1-1/(2*phi_1)),col2*(1-1/(2*phi_2)))
# theo_chi1= mean(apply(tmp,1, min)) 
# theo_chi2= mean(apply(tmp,1, max)) 
# v1 <- (w1*gamma)^{1/2}/sum((w1*gamma)^{1/2})
# v2 <- (w2*gamma)^{1/2}/sum((w2*gamma)^{1/2})
# lower_bound <- theo_chi1*sum(apply(cbind(v1,v2),1,min))
# upper_bound <- theo_chi2*sum(apply(cbind(v1,v2),1,max))
# 

## Numerical limits
load("~/Desktop/Nonstat/Sec4_figures/Realizations_different.RData")
Ecdf_tmp <- apply(Realizations, 2, ecdf)
# Ecdf_tmp9 <-ecdf(Realizations[,1])
# Ecdf_tmp8 <-ecdf(Realizations[,2])
# Ecdf_tmp6 <-ecdf(Realizations[,3])

Margin_Unif = matrix(NA, nrow(Realizations), ncol(Realizations))
for(i in 1:ncol(Realizations)){
  Margin_Unif[,i] = Ecdf_tmp[[i]](Realizations[,i])
}
# Margin_Unif = cbind(Ecdf_tmp9(Realizations[,1]), Ecdf_tmp8(Realizations[,2]),
#                     Ecdf_tmp6(Realizations[,3]))  # Marginal Uniform Scale
u = c(seq(0.9,0.9937895,length.out = 30),seq(0.995,0.99999,length.out = 30))
# u = c(seq(0.9,0.9937895,length.out = 39),seq(0.995,0.99999,length.out = 50))
Chi = matrix(NA,nrow=length(u), ncol=6)
for(i in 1:length(u)){
  if(i %% 10==0) cat('--',i,'\n')
  Chi[i,1]=mean(apply(Margin_Unif,1,function(x) x[9]>u[i]&x[6]>u[i]))/(1-u[i]) #AD
  Chi[i,2]=mean(apply(Margin_Unif,1,function(x) x[9]>u[i]&x[5]>u[i]))/(1-u[i]) #AD 
  Chi[i,3]=mean(apply(Margin_Unif,1,function(x) x[7]>u[i]&x[8]>u[i]))/(1-u[i]) #AD weak
  Chi[i,4]=mean(apply(Margin_Unif,1,function(x) x[1]>u[i]&x[2]>u[i]))/(1-u[i]) #AI
  Chi[i,5]=mean(apply(Margin_Unif,1,function(x) x[2]>u[i]&x[3]>u[i]))/(1-u[i]) #AI stronger or weaker than previous
  Chi[i,6]=mean(apply(Margin_Unif,1,function(x) x[1]>u[i]&x[4]>u[i]))/(1-u[i]) #AI 
}
save(Chi, file='~/Desktop/Nonstat/Sec4_figures/Different_radius1.RData')


## ********************************
## ********** VISUALIZE ***********
## ********************************
load('~/Desktop/Nonstat/Sec4_figures/Different_radius.RData')
u = c(seq(0.9,0.9937895,length.out = 39),seq(0.995,0.99999,length.out = 50))
plot(u, Chi[,1], type='l', col=1, lwd=2, ylim=c(0,0.2))
points(u, Chi[,2], type='l', col=2, lwd=2)
points(u, Chi[,3], type='l', col=3, lwd=2)
# *In the same radius, give you higer dependence even if phi<0.5*
points(u, Chi[,4], type='l', col=4, lwd=2)
points(u, Chi[,5], type='l', col=5, lwd=2) 

# ----------------------
# ---- Point 6 vs 9 ----
# ----------------------
ind = 1
smoothingSpline = smooth.spline(u[40:length(u)], Chi[40:length(u),ind], spar=0.98)
plot(smoothingSpline,lwd=2)
lines(smoothingSpline,lwd=2, col='blue')

## Theoretical bounds
phi_1 <- phi_vec[9]; phi_2 <- phi_vec[6]
col1 = W_Realizations[,9]^{1/(2*phi_1)}
col2 = W_Realizations[,6]^{1/(2*phi_2)}
tmp = cbind(col1/mean(col1),col2/mean(col2))
theo_chi1= mean(apply(tmp,1, min)) 
theo_chi2= mean(apply(tmp,1, max)) 
w1 <- Weight_matrix[9,]; w2 <- Weight_matrix[6,]
v1 <- (w1*gamma)^{1/2}/sum((w1*gamma)^{1/2})
v2 <- (w2*gamma)^{1/2}/sum((w2*gamma)^{1/2})
lower_bound <- theo_chi1*sum(apply(cbind(v1,v2),1,min))
upper_bound <- theo_chi2*sum(apply(cbind(v1,v2),1,max))


chi_u_sequence <- data.frame(u=u,chi=c(Chi[1:39,ind], smoothingSpline$y))
limit_chi <- data.frame(u=1,chi=0.07783528)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_bound),linetype='dashed', colour='red',size=1.2)+
  geom_hline(yintercept=lower_bound,linetype='dashed', colour='red',size=1.2)+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='red')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,6])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario2_96.pdf', width=3,height=3)
print(plot_gg)
dev.off()

# eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
# N =30000000
# rho = Cov[9,6]
# Finverse<-function(u) {return(1/(1-u))}
# Realizations<-matrix(NA,nrow=N,ncol=2)
# W_Realizations<-matrix(NA,nrow=N,ncol=2)
# Cov <- matrix(c(1,rho,rho,1),ncol=2)
# cat('When phi_1=',phi_1,', phi_2=',phi_2,', and rho=',rho,':\n')
# for(j in 1:N){
#   if(j %% 10000==0) cat('--',j,'\n')
#   Z<-mvtnorm::rmvnorm(1, sigma = Cov)
#   W<-Finverse(pnorm(Z))
#   S <- rlevy(length(w1), m=0, s=gamma)
#   R1<-sum(w1*S)
#   R2<-sum(w2*S)
# 
#   X<-c(R1^phi_1*W[1],R2^phi_2*W[2])
#   Realizations[j,]<-X
#   W_Realizations[j,] <- W
# }

Ecdf_tmp1 <-ecdf(Realizations[,9])
Ecdf_tmp2 <-ecdf(Realizations[,6])
Margin_Unif = cbind(Ecdf_tmp1(Realizations[,9]), Ecdf_tmp2(Realizations[,6]))  # Marginal Uniform Scale

q<-seq(0.000001,0.1,length.out = 300)
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q,1/eta_inverse,type='l',xlim=c(0.9,1),ylim=c(0,1),
     xlab='1-p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[9,6])))
grid()

smoothingSpline = smooth.spline(1-q[2:10], 1/eta_inverse[2:10], spar=0.45)
plot(smoothingSpline,lwd=2, xlim=c(0.995,1), ylim=c(0.7,1))
lines(smoothingSpline,lwd=2, col='blue')
points(1-q[1:10], 1/eta_inverse[1:10], col='red', pch=20)


eta_u_sequence <- data.frame(u=c(1,1-q),
              eta=c(1,1/eta_inverse[1], rev(smoothingSpline$y),1/eta_inverse[11:length(q)]))
limit_eta <- data.frame(u=1, eta=1)
plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_line(colour='#4c3480', size=0.9) +
  geom_point(data = limit_eta, mapping = aes(x=u, y=eta), size=1, colour='red')+
  # geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,6])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.25,1.02),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))

pdf('~/Desktop/Nonstat/Sec4_figures/scenario2_96_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()


# ----------------------
# ---- Point 9 vs 5 ----
# ----------------------
ind = 2
smoothingSpline = smooth.spline(u[40:length(u)], Chi[40:length(u),ind], spar=0.8)
plot(smoothingSpline,lwd=2)
lines(smoothingSpline,lwd=2, col='blue')
points(u[40:length(u)], Chi[40:length(u),ind],pch=20,col='red')

chi_u_sequence <- data.frame(u=u,chi=c(Chi[1:39,ind], smoothingSpline$y))
limit_chi <-  data.frame(u=1,chi=0)
plot_gg <- ggplot(chi_u_sequence, aes(x=u, y=chi)) + geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 0, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[5],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[5], Cov[9,5])), 
       y=expression(chi(u)))+
  scale_y_continuous(limits=c(-0.01,0.5),expand = c(0,0)) + scale_x_continuous(limits=c(min(u),1.0018),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))
pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_95.pdf', width=3,height=3)
print(plot_gg)
dev.off()

## eta estimation: log(P(F1(X1)>1-q, F2(X2)>1-q))/log(q) -> 1/eta_X
alpha <- 1/2
eta_W <- (1+Cov[9,5])/2
phi_1 <- phi_vec[9]; phi_2 <- phi_vec[5] ## no shared non-zero weights
# phi_1/(alpha*eta_W) < 2
rho=Cov[9,5]
lower_eta <- alpha*eta_W/phi_1; upper_eta <- alpha/phi_2

N =5e+07

Realizations_95<-matrix(NA,nrow=N,ncol=2)
W_Realizations_95<-matrix(NA,nrow=N,ncol=2)
cat('When phi1=',phi_1,', phi2=',phi_2,', and rho=',rho,':\n')
weights_matrix <- Weight_matrix[c(9,5),]
cov <- matrix(c(1,rho,rho,1),ncol=2)
for(j in 1:N){
  if(j %% 1000000==0) cat('--',j,'\n')
  Z<-mvtnorm::rmvnorm(1, sigma = cov)
  W_vec<-Finverse(pnorm(Z))
  S <- rlevy(nrow(Knots_data), m=0, s=gamma)
  R_vec <- apply(weights_matrix,1, function(x) sum(x*S))
  
  X<-R_vec^c(phi_1,phi_2)*W_vec
  Realizations_95[j,]<-X
  W_Realizations_95[j,] <- W_vec
}

Ecdf_tmp1 <-ecdf(Realizations_95[,1])
Ecdf_tmp2 <-ecdf(Realizations_95[,2])
Margin_Unif = cbind(Ecdf_tmp1(Realizations_95[,1]), 
                    Ecdf_tmp2(Realizations_95[,2]))  # Marginal Uniform Scale

# Ecdf_tmp1 <-ecdf(Realizations[,9])
# Ecdf_tmp2 <-ecdf(Realizations[,5])
# Margin_Unif = cbind(Ecdf_tmp1(Realizations[,9]), Ecdf_tmp2(Realizations[,5]))  # Marginal Uniform Scale

q<-seq(0.00004,0.1,length.out = 400)
Min_unif<-apply(Margin_Unif,1,function(x) min(x))
eta_inverse <- rep(NA,length(q))
for(i in 1:length(q)){
  eta_inverse[i]<-log(mean(Min_unif>1-q[i]))/log(q[i])
}

plot(1-q[-1],1/eta_inverse[-1],type='l',xlim=c(0.9,1),ylim=c(0.5,0.6),
     xlab='1-p',ylab=expression(eta[X]),
     main=parse(text=sprintf("paste(phi[1],'=',%.2f,', ',phi[2],'=',%.2f, ', ' ,rho,'=',%.1f)",phi_1,phi_2,Cov[9,6])))
abline(h=lower_eta, lty=2, col='red', lwd=2)
abline(h=upper_eta, lty=2, col='red', lwd=2)
grid()

smoothingSpline = smooth.spline(1-q[c(3,6:140)], 1/eta_inverse[c(3,6:140)], spar=0.65)
plot(smoothingSpline,lwd=2, ylim=c(0.50,0.5279639))
lines(smoothingSpline,lwd=2, col='blue')
points(1-q[c(2,6:140)], 1/eta_inverse[c(2,6:140)], col='green', pch=20)

smoothingSpline2 = smooth.spline(1-q[140:length(q)], 1/eta_inverse[140:length(q)], spar=0.85)
plot(smoothingSpline2,lwd=2, ylim=c(0.5165, 0.5279639), xlim=c(0.9,1))
lines(smoothingSpline2,lwd=2, col='blue')
points(1-q, 1/eta_inverse, col='red', pch=20)

eta_u_sequence <- data.frame(u=c(smoothingSpline2$x, smoothingSpline$x, 1),eta=c(smoothingSpline2$y, smoothingSpline$y, 0.5250466))
eta_u_sequence <- eta_u_sequence[c(1:344, 398),]
save(eta_inverse, eta_u_sequence, file='~/Desktop/Nonstat/Sec4_figures/eta_95.RData')

plot_gg <- ggplot(eta_u_sequence, aes(x=u, y=eta)) + 
  geom_hline(yintercept=lower_eta,linetype='dashed', colour='red',size=1.2)+
  geom_line(colour='#4c3480', size=0.9) +
  geom_hline(yintercept = 1, linetype='dashed') + geom_vline(xintercept = 1, linetype='dashed')+
  geom_hline(yintercept=min(0.999,upper_eta),linetype='dashed', colour='red',size=1.2)+
  # geom_point(data = limit_chi, mapping = aes(x=u, y=chi), size=1, colour='blue')+
  labs(title=parse(text=sprintf("paste(phi[9],'=',%.3f,', ',phi[6],'=',%.3f, ', ' ,rho,'=',%.2f)",phi_vec[9],phi_vec[6], Cov[9,5])), 
       y=expression(eta(u)))+
  scale_y_continuous(limits=c(0.5,0.6),expand = c(0,0)) + scale_x_continuous(limits=c(min(1-q),1.0015),expand = c(0,0)) +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_line(colour = "gray", linetype = "dotted"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA))

pdf('~/Desktop/Nonstat/Sec4_figures/scenario1_95_eta.pdf', width=3,height=3)
print(plot_gg)
dev.off()




