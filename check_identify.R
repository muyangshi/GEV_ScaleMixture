load('C_mu0_ro.gzip')
load('C_mu1_ro.gzip')
load('mu_matrix_ro.gzip')
load('Beta_mu0_ro.gzip')
load('Beta_mu1_ro.gzip')

# Design Matrix
C_mu0 <- rbind(t(C_mu0_ro[,,1]))
for(t in 2:16){
  C_mu0 <- rbind(C_mu0, t(C_mu0_ro[,,t]))
}

Time <- c(-8:7)
C_mu1 <- rbind(t(C_mu1_ro[,,1]) * Time[1])
for(t in 2:16){
  C_mu1 <- rbind(C_mu1, t(C_mu1_ro[,,t]) * Time[t])
}

X <- cbind(C_mu0, C_mu1)

# Response
y <- mu_matrix_ro[,1]
for(t in 2:16){
  y <- c(y, mu_matrix_ro[,t])
}
y <- y + rnorm(50*16, mean = 0, sd = 0.01)

mod <- lm(y ~ X-1)
summary(mod)

mod$coefficients
c(Beta_mu0_ro, Beta_mu1_ro)

# mod$coefficients mostly matches with preset Betas
# no obvious issue of identifiability
