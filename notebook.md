# Meeting Summaries on GEV Project

## Sept 24 Tuesday Meeting with Likun/Ben

### Poster

- [ ] Ben's "p-card" -- purchasing card?

- Section 2:
  - [x] No color fill for equations. Just a box. It's too popped.
  - [x] Non-stationarity is for $\phi(s)$ and $\rho(s)$;  Scale-awareness means the local AI or AD, long range AI.
- Section 3 Simulation:
  - [x] Figure 1 caption references the parameter surfaces, not the data
  - [x] Use the actual parameter surface not the Likun's
- Section 3 Data Application:
  - [x] Shorter Intro, remove computation even (to make space)
  - [x] Make space to put data $\chi$ and fitted $\chi$ side-by-side
  - [x] Modify figure 5 caption: comparing model realization chi versus empirical dataset one
- Section 4 Discussion:
  - [x] no s
  - [x] Make bulletpoints
    - [x] no two step -- phrased in a positive way: put the marginal part integrated into the model
    - [x] AI $\phi(s)$ but spatially varying
    - [x] ToDo: GPD
- [x] Remove ASA Logo, put on NSF logo


### Paper

Went over changes made on plots: 
  - Figure 3 the Thm2.3 simulation plot
  - Current stage of the Figure 4 (empirical estimates of $\chi$ and $\eta$)
  - the simulation scenarios plots 3D -> 2D
  - the data application posterior single hue plots
  - The moving window $\chi$-plots are good as they are now.
  
#### Todo

- [ ] Throught the paper, change "marginal distribution of the copula" to the dependence model
- [ ] Update model names in the manuscript texts
- [ ] Follow Ben's edits

### Simulation $\chi$ and $\eta$ - Example 3 Figure 4

- Computation:
  - [x] save and load data
  - [x] fix the bug in $\eta$ estimation and redo estimations
  - [x] eventually use $N = 300,000,000$ datapoints
  - [ ] Using `mev` in `R` to estimate $\eta$ because treating $L(1-u)$ as 1 may be biased
  - Likun: limit of $\eta$ can be calculated by transforming to unit Frechet and fit GPD: 
    ```##Calculate the tail dependence using min(Xi,Xj)
    T1<-rank(Realizations[,1])/(N+1)
    T2<-rank(Realizations[,2])/(N+1)
    NewReal<-data.frame(xi=FrechetInv(T1), xj=FrechetInv(T2))
    Tmin<-apply(NewReal,1,min)
    fit<-gpd.fit(Tmin,method='amle')
    itaX<-fit[1]
    library(mvtnorm)
    library(EnvStats)
    library(gPdtest)
    ```
  
- Plotting/aesthetics:
  - [x] Place UB above LB in legend
    - `tab:red` solid line represent theoretical limit
    - `tab:orange` dashed line represent UB
    - `tab:blue` dotted line represent LB
  - [x] the circle may not be necessary
  - [x] different colors and linestyle for the UB and LB
  - [x] thicker lines for the lines drawn first (if overlapping)
  - [ ] if $\chi$ is not smooth, use smoothing splines to smooth it



## Sept 17 Tuesday Meeting with Likun/Mark/Ben

### Paper - Ben will give a pass on Thursday

*Do this first, Ben will give a pass on Thursday*

- Section 2 Model
  - [x] Change the "marginal distribution of the copula" to univariate distribution of the dependence model -- **through out the paper!**
  - [ ] Remade Figure 3 ($\phi(s)$ surface) Figure 4 (empirical $\chi$ and $\eta$) so that we are not making radius able to change and have plots verifying the bounds in Thm 2.3
    - [x] Figure 3 - simulation case
    - [ ] Figure 4 - empirical plots
- Section 3 Bayesian Inference
  - [x] add hierarchical model into section 3
  - [x] paragraph on knot (spatial dimension reduction) for the dependence model (don't say process model because levels are not clear)
  - [x] move the splines about marginal data model to section 5
- Section 4 Simulation Study
  - [x] Figure 3 has already used scenario. Change simulation scenario to case. (Maybe unneccessary as we will redo Figure 3 with just one scenario)
  - [x] Plotting
    - [x] Make 2D heat map for simulatoin scenarios, instead of 3D `plot_simulation_surface.py`
    - [x] Larger axis labels for the coverage plots `coverage_analysis.py`
- Section 5 Data Application
  - [x] Add comments for moving window $\chi$-plot. Comment on which regions and which plots are matching (captured asymptotic dependence). Mention that model realization has gradually fading color indicating AI (which is matches with the inferenced $\phi(s)$ surface `Note the error/bug in qRW(u, phi, rho)`
  - [x] Plotting: 
    - [x] Change the $\phi(s)$ surface plot so that 0.5 is white
    - [x] Change the $\rho(s)$ surface plot to single color
    - [x] Potentially also changing the $\theta_{GEV}(s)$ plots to have one hue?
    - [x] Change the bound of colorbar for $\chi$-plot to [0, 0.5]. 
  - [ ] Update model names in the manuscript texts

### Re-do $\eta$ and $\chi$ empricial estimation for Thm 2.3

- [ ] Empirically estimate $\chi$ and $\eta$
  - Treat $L_Z(...)$ as 1 since we are approaching the limit. Likun: do a regression on (log of) the formula to get $\eta$
  - Roughly 2 hours for $N = 300,000,000$ to get six $\chi$ and $\eta$; doing $N = 100,000,000$ seems to be much faster.
  - $u$ goes to 0.999999 not enough for (a) (iii)'s $\chi$ (with $N = 100,000,000$)? Likun used cubic spline smooth, I'm plotting the raw numerical estimate
  - [x] empirical $\chi$
  - [x] empirical $\eta$
  - [x] bound for $\chi$
  - [x] bound for $\eta$
    - $\eta^W_{ij}$ appears to be (1 + $\rho_{ij}$)/2 for Normal distribution?
- $\eta$ fits in the story-telling because there are bounds in the Theorem (red dashed lines), and empirical result can show "they are good bounds"
- **Same** simulation with special placements of the points should be sufficient; we also don't want to change the radii. 
- Chien-Chung used https://cran.r-project.org/web/packages/mev/mev.pdf to estimate $\eta$


### Some documents on choosing colormaps:
- https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html
- https://colorbrewer2.org/#type=sequential&scheme=OrRd&n=3
