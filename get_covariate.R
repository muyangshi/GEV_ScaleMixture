library(plotly)
library(mgcv)

# load
load('gs_x_ro.gzip')
load('gs_y_ro.gzip')
load('gs_xy_ro.gzip')
load('mu_surf_grid_ro.gzip')

load('sites_xy_ro.gzip')
load('mu0_1t_ro.gzip')

gs_xy_df <- as.data.frame(gs_xy_ro)
colnames(gs_xy_df) <- c('x','y')

sites_xy_df <- as.data.frame(sites_xy_ro)
colnames(sites_xy_df) <- c('x','y')

mu_surf_grid_df <- data.frame(x=double(),
                              y=double(),
                              mu=double())
for(row in 1:length(gs_x_ro)){
  for(col in 1:length(gs_y_ro)){
    x_cor <- gs_x_ro[row]
    y_cor <- gs_y_ro[col]
    mu_surf_grid_df[nrow(mu_surf_grid_df) + 1,] = c(x_cor,y_cor,mu_surf_grid_ro[row,col])
  }
}

basis <- smoothCon(s(x, y, k = 12, fx = TRUE), data = gs_xy_df)[[1]]
X_site <- PredictMat(basis, data = sites_xy_df)
beta <- coef(lm(c(mu0_1t_ro) ~ X_site-1))
persp(gs_x_ro, gs_y_ro, mu_surf_grid_ro,
      theta = -30, ticktype = 'detailed',
      main = 'true surface', xlab = 'x', ylab = 'y', zlab = 'mu') # true surface
# persp(gs_x_ro, gs_y_ro, matrix(mu_surf_grid_df$mu, nrow = 41, byrow = TRUE),
#       theta = -30, ticktype = 'detailed',
#       main = 'true surface', xlab = 'x', ylab = 'y', zlab = 'mu') # true surface
persp(gs_x_ro, gs_y_ro, matrix(c(basis$X %*% beta), nrow = 41, ncol = 41, byrow=TRUE),
      theta = -30, ticktype = 'detailed',
      main = 'using observed data', xlab = 'x', ylab = 'y', zlab = 'mu')

# Note that X_site is different from X_site_direct
# X_site_direct <- smoothCon(s(x, y, k = 30, fx=TRUE), data = sites_xy_df)[[1]]$X

my_k <- 12
basis <- smoothCon(s(x, y, k = my_k, fx = TRUE), data = gs_xy_df)[[1]]
beta <- rep(0.1, my_k)
persp(gs_x_ro, gs_y_ro, matrix(c(basis$X %*% beta), nrow = 41, ncol = 41, byrow=FALSE),
      theta = -30, ticktype = 'detailed',
      main = 'using observed data', xlab = 'x', ylab = 'y', zlab = 'mu')

persp(gs_x_ro, gs_y_ro, matrix(basis$X[,c(10)], nrow = 41, ncol = 41, byrow=FALSE),
      theta = -30, ticktype = 'detailed',
      main = 'using observed data', xlab = 'x', ylab = 'y', zlab = 'mu')



xy <- data.frame(x = double(), y = double())
for(row in 1:100){
  for(col in 1:100){
    xy[nrow(xy)+1,] = c(x[row], y[col])
  }
}

basis1 <- smoothCon(s(x, y, k = 12, fx = TRUE), data = gs_xy_df)[[1]]
basis2 <- smoothCon(s(x, y, k = 12, fx = TRUE), data = gs_xy_df)[[1]]

################################################################################

# Spline from grids xy
basis_grid <- smoothCon(s(x, y, k = 100, fx=FALSE), data = gs_xy_df)[[1]]
basis_site <- smoothCon(s(x, y, k = 100, fx=FALSE), data = sites_xy_df)[[1]]

X_grid_basis_grid <- PredictMat(basis_grid, data = gs_xy_df) # same as basis_grid$X
X_grid_basis_site <- PredictMat(basis_site, data = gs_xy_df) # interpolate site splines to grid
X_site_basis_grid <- PredictMat(basis_grid, data = sites_xy_df) # interpolate grid splines to sites
X_site_basis_site <- PredictMat(basis_site, data = sites_xy_df) # same as basis_site$X

beta_cheat <- coef(lm(mu_surf_grid_df$mu ~ X_grid_basis_grid-1)) # regression spline fit using true surface (WHICH WE DONT HAVE)
beta_nocheat <- coef(lm(c(mu0_1t_ro) ~ X_site_basis_grid-1)) # regression spline fit using observed data

beta_nocheat2 <- coef(lm(c(mu0_1t_ro) ~ X_site_basis_site-1))

persp(gs_x_ro, gs_y_ro, mu_surf_grid_ro,
      theta = -30, ticktype = 'detailed',
      main = 'true surface', xlab = 'x', ylab = 'y', zlab = 'mu') # true surface
persp(gs_x_ro, gs_y_ro, matrix(c(X_grid_basis_grid %*% beta_cheat), nrow = 41, ncol = 41, byrow = TRUE),
      theta = -30, ticktype = 'detailed',
      main = 'use true surface to fit basis_grid', xlab = 'x', ylab = 'y', zlab = 'mu')
persp(gs_x_ro, gs_y_ro, matrix(c(X_grid_basis_grid %*% beta_nocheat), nrow = 41, ncol = 41, byrow=TRUE),
      theta = -30, ticktype = 'detailed',
      main = 'use observed data to fit basis_grid', xlab = 'x', ylab = 'y', zlab = 'mu')

persp(gs_x_ro, gs_y_ro, matrix(c(X_grid_basis_site %*% beta_nocheat2), nrow = 41, ncol = 41, byrow=TRUE),
      theta = -30, xlab = 'x', ylab = 'y', zlab = 'mu')

hist(c(X_site_basis_grid %*% beta_cheat) - c(mu0_1t_ro)) # cheated beta
hist(c(X_site_basis_grid %*% beta_nocheat) - c(mu0_1t_ro)) # not cheated


observed_df <- cbind(sites_xy_df, c(mu0_1t_ro))
colnames(observed_df) <- c('x','y','mu')
gamfit <- gam(mu ~ s(x, y, fx = FALSE), data = observed_df)
vis.gam(gamfit)
scatterplot3d(x = observed_df$x, y = observed_df$y, z = observed_df$mu)



Xp <- PredictMat(basis, data = sites_xy_df) # "interpolate" to splines at sites (what we need in sampler)
hist(c(Xp %*% beta_nocheat) - c(mu0_1t_ro))


# What Ben suggests
# we can't create more splines than there are observations -- must need more than 100 sites
basis_sites <- smoothCon(s(x, y, k = 100, fx=TRUE), data = sites_xy_df)[[1]]
Xp_sites <- PredictMat(basis_sites, data = sites_xy_df) # same as basis_sites$X
beta_sites <- coef(lm(c(mu0_1t_ro) ~ basis_sites$X-1))
hist(c(Xp_sites %*% beta_sites) - mu0_1t_ro)

Xp_grid <- PredictMat(basis_sites, data = gs_xy_df)

persp(gs_x_ro, gs_y_ro, mu_surf_grid_ro,
      theta = -30)
persp(gs_x_ro, gs_y_ro, matrix(c(Xp_grid %*% beta_sites), nrow = 41, ncol = 41, byrow = TRUE),
      theta = -30)













################################################################################
# Plotting
# library(rgl)
# persp3d(gs_x_ro, gs_y_ro, gaussian_surface_ro)
library(plotly)
plot_ly() %>%
  add_trace(data = mu_surf_grid_df, 
            x = mu_surf_grid_df$x, 
            y = mu_surf_grid_df$y, 
            z = mu_surf_grid_df$mu, type='mesh3d')

# Spline Fitting
library(mgcv)
gamfit <- gam(mu ~ s(x, y, k = 40, fx=FALSE), data = mu_surf_grid_df) # fx=TRUE means fixed d.f. regression spline
# fitted spline surface
plot_ly() %>%
  add_trace(x = mu_surf_grid_df$x,
            y = mu_surf_grid_df$y,
            z = fitted(gamfit), type = 'mesh3d')
vis.gam(gamfit)
# residual surface
plot_ly() %>%
  add_trace(x = mu_surf_grid_df$x,
            y = mu_surf_grid_df$y,
            z = fitted(gamfit) - mu_surf_grid_df$mu, type = 'mesh3d')

# Extract the Splines(?)
# Check this with Ben and Likun!!!
Xp <- predict(gamfit, mu_surf_grid_df, type='lpmatrix')
matrix_product <- Xp %*% coef(gamfit)
matrix_product - fitted(gamfit)

################################################################################
# Another way: directly constructing smoothCon object
library(mgcv)
xy_df <- mu_surf_grid_df[,c(1:2)]
basis <- smoothCon(s(x, y, k = 100, fx=TRUE), data = xy_df)[[1]]
# Xp2 <- PredictMat(basis, data = mu_surf_grid_df[c(1:30),]) #?
beta <- coef(lm(mu_surf_grid_df$mu~basis$X-1)) # X-1 removes the intercept. Vignette does this regression spline model.
Xp <- PredictMat(basis, data = xy_df)

# plot the Xp basis (columns)
plot_ly() %>%
  add_trace(x = mu_surf_grid_df$x,
            y = mu_surf_grid_df$y,
            z = Xp[,20], type = 'mesh3d')
persp(gs_x_ro, gs_y_ro, matrix(data = Xp[,40], nrow = 11, ncol=11, byrow=TRUE),
      theta = -45,
      xlab = 'x', ylab = 'y', zlab = 'b', ticktype = 'detailed')
persp(gs_x_ro, gs_y_ro, matrix(data = Xp %*% beta, nrow = 11, ncol=11, byrow=TRUE),
      theta = -45,
      xlab = 'x', ylab = 'y', zlab = 'b', ticktype = 'detailed')

smoothCon()

Xp2 %*% beta2 - mu_surf_grid_df$mu


new_x <- seq(0.5, 10, length.out=8)
new_y <- seq(4, 6, length.out=3)
new_df <- data.frame(x=double(),
                     y=double())
for(row in 1:length(new_x)){
  for(col in 1:length(new_y)){
    new_df[nrow(new_df) + 1,] = c(new_x[row],new_y[col])
  }
}

smoothCon(s(x, y, k = 100, fx=TRUE), data = new_df)[[1]]

Xp3 <- PredictMat(basis, data = new_df)

new_new_df <- rbind(mu_surf_grid_df[,c(1,2)], new_df)

new_basis <- smoothCon(s(x, y, k = 100, fx=TRUE), data = new_new_df)[[1]]

last24 <- new_basis$X[c(1072:1095),]

Xp3 - last24

dim(PredictMat(basis, data = new_df))

# plot(gamfit, pages=1, residuals=TRUE, seWithMean=TRUE)
# plot(gamfit)
# predict(gamfit, type='response')
# predict(gamfit, type='lpmatrix')
# predict(gamfit, typle = 'terms')
# gamfit[["smooth"]][[1]][["S"]][[1]]
# fitted(gamfit)
# coef(gamfit)



# jagam -- what Likun uses(?)
# jagamfit <- jagam(mu ~ s(x, y, k = 100), data = mu_surf_grid_df)
# require rjags
# jags.file <- "/test.jags"
out$jags.X

################################################################################
# Why removing the (Intercept) from lm is done by adding -1?  ##################
################################################################################
#   Linear regression is of the form
# y = mx + b
# right?
#   And in R, - means omit, as in
# mydataframe[, -1]
# right?
#   But when you specify a formula within lm(), the intercept is implicit.
# That is, you write:
#   y ~ x
# and m and b are fitted.
# So if you want to omit the intercept, you use 1 as a placeholder
# rather than leaving the - dangling somewhere.
# y ~ x - 1
# But as you say, there are other ways, so use the one you like.
# Note that if you really wanted to subtract 1 from x before fitting the
# model, you'd need to make that clear to R:
# y ~ I(x - 1)
# This is all in the help for formula, where it says "The - operator
# removes the specified terms".
# > So I found out that to remove the (Intercept) term from lm's model one can
# > add -1 to the predictors. I.e. do lm(resp ~ x1 + x2 - 1)
# > Another way is to add 0, e.g. lm(resp ~ 0 + x1 + x2).