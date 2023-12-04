library(gap) # for qqunif
library(rmutil) # for levy
library(ggplot2)

random_levy = rlevy(n = 1000, m = 0, s = 1)
hist(plevy(random_levy, m = 0, s = 1))

# scale = 1
random_levy1 = rlevy(n = 1000, m = 0, s = 1)
qqunif(plevy(random_levy1, m = 0, s = 1))
random_levy2 = rlevy(n = 1000, m = 0, s = 1)
random_levy3 = rlevy(n = 1000, m = 0, s = 1)
random_levy4 = rlevy(n = 1000, m = 0, s = 1)
random_levy5 = rlevy(n = 1000, m = 0, s = 1)
random_levy6 = rlevy(n = 1000, m = 0, s = 1)
random_levy7 = rlevy(n = 1000, m = 0, s = 1)
random_levy8 = rlevy(n = 1000, m = 0, s = 1)
random_levy9 = rlevy(n = 1000, m = 0, s = 1)
random_levy_sum = random_levy1 + random_levy2 + random_levy3 + 
  random_levy4 + random_levy5 + random_levy6 + random_levy7 +
  random_levy8 + random_levy9
# hist(plevy(random_levy_sum, m = 0, s = 9^2))
qqunif(plevy(random_levy_sum, m = 0, s = 9^2), logscale=FALSE)

# scale = 0.5
random_levy1 = rlevy(n = 1000, m = 0, s = 0.5)
qqunif(plevy(random_levy1, m = 0, s = 0.5))
random_levy2 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy3 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy4 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy5 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy6 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy7 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy8 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy9 = rlevy(n = 1000, m = 0, s = 0.5)
random_levy_sum = random_levy1 + random_levy2 + random_levy3 + 
  random_levy4 + random_levy5 + random_levy6 + random_levy7 +
  random_levy8 + random_levy9
# hist(plevy(random_levy_sum, m = 0, s = 9^2))
qqunif(plevy(random_levy_sum, m = 0, s = 0.5*9^2), logscale=FALSE)

x <- seq(0.01, 10, 0.05)
plot(x, dlevy(x, m = 0, s = 1), type = 'l', ylim=c(0,1))
lines(x, dlevy(x, m = 0, s = 0.5), color = 'blue')


qqunif(plevy(0.5*rlevy(n = 1000, m = 0, s = 1), m = 0, s = 0.5), logscale=FALSE)

qqunif(pnorm(0.5*rnorm(1000,mean=0,sd=2), mean = 0, sd=1), logscale=FALSE)

