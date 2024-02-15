# Isometric grid
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
isometric_grid <- isometric_grid[isometric_grid$x < xmax,]
plot(isometric_grid, asp = 1, pch = "+", cex = 1.5, xlim = c(0,10), ylim = c(0,10))
abline(v = c(0,10), h = c(0,10), col = "gray")
for(g in 1:nrow(isometric_grid)) plotrix::draw.circle(x = isometric_grid[g,1], y = isometric_grid[g,2], radius = 2.5, border = 2)
