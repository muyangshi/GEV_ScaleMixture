library(extRemes)
library(ordinal)
load('pY_ro.gzip')
load('pY_mcmc_ro.gzip')
load('pY_smooth_ro.gzip')
load('gumbel_pY_ro.gzip')
load('gumbel_pY_mcmc_ro.gzip')
load('gumbel_pY_smooth_ro.gzip')
pY               <- pY_ro
pY_mcmc          <- pY_mcmc_ro
pY_smooth        <- pY_smooth_ro
gumbel_pY        <- gumbel_pY_ro
gumbel_pY_mcmc   <- gumbel_pY_mcmc_ro
gumbel_pY_smooth <- gumbel_pY_smooth_ro

set.seed(2345)
Ns <- dim(pY_ro)[1]
Ns <- 590
###############################
# In-sample 590 Gumbel QQPlot #
###############################

# Single Site Gumbel QQPlot ---------------------------------------------------------------------------

## with MLE-fit marginal GEV parameters
s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s = sort(gumbel_pY[s,])
nquants = length(gumbel_s)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
# plot(gumbel_s, emp_q)
# abline(a = 0, b = 1)
qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                lwd=3)
pdf(file=paste("R_GEVfit-QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

## with per MCMC iteration fit marginal GEV parameters
# s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s_mcmc = sort(apply(gumbel_pY_mcmc[,s,],2, mean))
nquants = length(gumbel_s_mcmc)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
# plot(gumbel_s_mcmc, emp_q)
# abline(a = 0, b = 1)
qq_gumbel_s_mcmc <- extRemes::qqplot(gumbel_s_mcmc, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("Modelfit-QQPlot of Site:",s),
                                lwd=3)
pdf(file=paste("R_Modelfit-QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

## with initial MLE smoothed marginal GEV parameters 
s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s = sort(gumbel_pY_smooth[s,])
nquants = length(gumbel_s)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
# plot(gumbel_s, emp_q)
# abline(a = 0, b = 1)
qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                lwd=3)
pdf(file=paste("R_InitSmooth_QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

# Overall (site time) Gumbel QQPlot  ------------------------------------------------------------------

## with GEV-fit marginal parameters
gumbel_overall = sort(as.vector(gumbel_pY))
nquants = length(gumbel_overall)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
qq_gumbel_overall <- extRemes::qqplot(gumbel_overall, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main="GEVfit-QQPlot Overall",
                                lwd=3)
pdf(file="R_GEVfit-QQPlot_Overall.pdf", width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_overall$qdata$x, qq_gumbel_overall$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_overall$qdata$x, qq_gumbel_overall$qdata$y, pch=20)
lines(qq_gumbel_overall$qdata$x, qq_gumbel_overall$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_overall$qdata$x, qq_gumbel_overall$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

## with Copula Model-fit marginal parameters
gumbel_mcmc_overall = sort(as.vector(apply(gumbel_pY_mcmc, c(2,3), mean)))
nquants = length(gumbel_mcmc_overall)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
qq_gumbel_mcmc_overall <- extRemes::qqplot(gumbel_mcmc_overall, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main="Modelfit-QQPlot Overall",
                                lwd=3)
pdf(file="R_Modelfit-QQPlot_Overall.pdf", width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_mcmc_overall$qdata$x, qq_gumbel_mcmc_overall$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_mcmc_overall$qdata$x, qq_gumbel_mcmc_overall$qdata$y, pch=20)
lines(qq_gumbel_mcmc_overall$qdata$x, qq_gumbel_mcmc_overall$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_mcmc_overall$qdata$x, qq_gumbel_mcmc_overall$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

###############################
# On 1034-590 Holdout Dataset #
###############################

# Single Site Gumbel QQPlot ---------------------------------------------------------------------------
load('pY_holdout_ro.gzip')
load('gumbel_pY_holdout_ro.gzip')
Ns <- dim(pY_holdout_ro)[1]
pY_holdout        <- pY_holdout_ro
gumbel_pY_holdout <- gumbel_pY_holdout_ro

## with predicted Copula Model fit marginal parameters
s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s_holdout = sort(gumbel_pY_holdout[s,])
nquants          = length(gumbel_s_holdout)
emp_p            = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q            = qgumbel(emp_p)
# plot(gumbel_s_holdout, emp_q)
# abline(a = 0, b = 1)
qq_gumbel_s_holdout <- extRemes::qqplot(gumbel_s_holdout, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("QQPlot Copula Model Predicted Marginal of Site:",s),
                                lwd=3)
pdf(file=paste("R_ModelPred_QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s_holdout$qdata$x, qq_gumbel_s_holdout$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s_holdout$qdata$x, qq_gumbel_s_holdout$qdata$y, pch=20)
lines(qq_gumbel_s_holdout$qdata$x, qq_gumbel_s_holdout$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s_holdout$qdata$x, qq_gumbel_s_holdout$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

load('pY_mcmc_holdout_ro.gzip')
load('gumbel_pY_mcmc_holdout_ro.gzip')
pY_mcmc_holdout        <- pY_mcmc_holdout_ro
gumbel_pY_mcmc_holdout <- gumbel_pY_mcmc_holdout_ro

## with Copula Model fit marginal parameters
s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s_mcmc_holdout = sort(apply(gumbel_pY_mcmc_holdout[,s,],2, mean))
nquants = length(gumbel_s_mcmc_holdout)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
# plot(gumbel_s_mcmc_holdout, emp_q)
# abline(a = 0, b = 1)
qq_gumbel_s_mcmc_holdout <- extRemes::qqplot(gumbel_s_mcmc_holdout, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("ModelPred Itermean QQPlot of Site:",s),
                                lwd=3)
pdf(file=paste("R_ModelPred_IterMean_QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s_mcmc_holdout$qdata$x, qq_gumbel_s_mcmc_holdout$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s_mcmc_holdout$qdata$x, qq_gumbel_s_mcmc_holdout$qdata$y, pch=20)
lines(qq_gumbel_s_mcmc_holdout$qdata$x, qq_gumbel_s_mcmc_holdout$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s_mcmc_holdout$qdata$x, qq_gumbel_s_mcmc_holdout$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

#####################
# On Real Test data #
#####################
library(readr)
blockMax_JJA_centralUS_test <- read_csv("blockMax_JJA_centralUS_test.csv")
stations_test <- read_csv("stations_test.csv")
save(blockMax_JJA_centralUS_test, file='blockMax_JJA_centralUS_test.RData')
save(stations_test, file='stations_test.RData')
load('blockMax_JJA_centralUS_test.RData')
load('stations_test.RData')

load('pY_smooth_test_ro.gzip')
load('pY_mcmc_test_ro.gzip')
load('gumbel_pY_smooth_test_ro.gzip')
load('gumbel_pY_mcmc_test_ro.gzip')

pY_smooth_test        <- pY_smooth_test_ro
pY_mcmc_test          <- pY_mcmc_test_ro
gumbel_pY_smooth_test <- gumbel_pY_smooth_test_ro
gumbel_pY_mcmc_test   <- gumbel_pY_mcmc_test_ro

Ns <- 99
## with initial MLE smoothed marginal GEV parameters
s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s = sort(gumbel_pY[s,])
nquants = length(gumbel_s)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                lwd=3)
pdf(file=paste("R_GEVfit-QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()

## with per mcmc iter marginal GEV parameters
# s <- floor(runif(1, min = 1, max = Ns+1)) # note that in R index starts from 1
gumbel_s_mcmc = sort(apply(gumbel_pY_mcmc[,s,],2, mean))
nquants = length(gumbel_s_mcmc)
emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
emp_q = qgumbel(emp_p)
qq_gumbel_s_mcmc <- extRemes::qqplot(gumbel_s_mcmc, emp_q, regress=FALSE, legend=NULL,
                                     xlab="Observed", ylab="Gumbel", main=paste("Modelfit-QQPlot of Site:",s),
                                     lwd=3)
pdf(file=paste("R_Modelfit-QQPlot_Site_",s,".pdf", sep=""), width = 6, height = 5)
par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
abline(a=0, b=1, lty=3, col="gray80", lwd=3)
legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
dev.off()