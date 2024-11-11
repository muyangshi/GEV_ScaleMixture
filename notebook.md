# Meeting Summaries on GEV Project

## TODOs

- Run the stationary model until convergence
- Posterior summaries
  - QQplot of the out-of-sample sites
  - Predictive log-likelihood of those testing sites

## Nov. 5 Tuesday Meeting with Ben

- Make a double spaced version of the paper, and see how long it is

## Oct. 29 Tuesday Meeting with Likun/Mark/Ben

Paper:

- [Huser, Opitz, Thibaud 2017](<../../Research/Spatial Extreme/1 Papers/Bridging Asymptotic Independence and Dependence in Spatial Extremes Using Gaussian Scale Mixtures.pdf>)
- [Huser and Wadsworth 2019](<../../Research/Spatial Extreme/1 Papers/Modeling Spatial Processes with Unknown Extremal Dependence Class.pdf>)
- [Hazra et al.](<../../Desktop/Efficient Modeling of Spatial Extremes over Large Geographical Domains.pdf>)
- [Wadsworth 2024](<../../Research/Spatial Extreme/1 Papers/Wadsworth 2024 Spatial extremal modelling A case study on the interplay between margins and dependence.pdf>)


Work:

- [x] Remake $\chi_{12}$ plot with only the $\chi$ limit (only lower bound)

- [x] GEV $\mu_1$ unit on $Y_t$:
  - $t = \dfrac{\text{year} - 1986}{21.8}$
  - 1986 is the center of {1949, ..., 2023}, and 21.8 is the sample std.
  - `np.max(predmu1) * 100 / np.std(all_years, ddof=1)` = 4.81 mm
  - `np.min(predmu1) * 100 / np.std(all_years, ddof=1)` = -4.49 mm

- [x] Running a stationary model for comparison
  - `recv(20) failed: Connection reset by peer (104)`
  - all else seems to be fine
  - restarted on 20241030 just to be safe
    - mixing seems very fine.

## Oct. 26 Friday Meeting with Ben

- [x] Paper Result section, GEV $\mu_1$ unit on $Y_t$?
  - precipitation measured in mm, fill in the blank
  - Be sure of the scaling on the denominator

- [x] Fit a Huser-Wadsworth version Stationary Model on the dataset
  - `/GEV_ScaleMixture/data/20241029_realdata_stationary`
  - set there to be one knot and very very large radius
  - Note that $\phi$ starting value was set to 0.4; shouldn't matter.
  - run the chain until convergence
  - posterior summaries
    - QQplot
    - Predictive Log-likelihood


- Golden poster Ben suggested:
  - consider removing the model equations on the first page
  - remove the simulation study on the second page, add a $\chi$-plot

## Oct. 21 Tuesday Meeting with Likun/Mark/Ben

- SBSS deadline November 15
- Ben will justify $\gamma_k = 0.5$
  - Ben will review section 5 and 6
- Likun will go over the paper

- followed Ben's edits
  - added plots to intro

## Oct. 15 Tuesday Meeting with Likun/Ben

- Talk about:
  - `mev`
  - limit of $\eta$ using Frechet
  - Paper edits

- [x] Get rid of the last point in $\eta_{34}$ -- call it numerical instability
- Ben will justify $\gamma_k = 0.5$
  - Ben will review section 5 and 6
- Likun will go over the paper on Friday

## Oct. 8 Tuesday Meeting with Likun/Mark/Ben

- Talk about `mev` results
  - Using the `betacop` estimator is just like a smooth, and doesn't improve estimation
  - `hill` with Exponential margin seems fine for AD
  - But we don't want inconsistency of switching between methods
- Check with Likun on Introduction section, proposition 2.1, and Appendix A
- Ben will look at the unresolved comments and Section 4, 5, & 6.
- Wadsworth's Paper on [marginal transformation influence dependence model](<../../Research/Spatial Extreme/1 Papers/Wadsworth Interplay between margins and dependence.pdf>)
- If time permits, start to [Work on GPD Model](../GPD_ScaleMixture/notebook.md)

### Work

- [x] Finish `mev` results with the hill estimator on Exponential margin
  - bug in earlier code; now use `CDF[,i:j]` instead of `CDF[,c(i,j)]`
  - [x] fix and re-run $N = 100,000,000$
    - results look good
  - [x] try to use $N = 300,000,000$ to match with $\chi$
    - $\eta_{34}$ is not good (sharing a kernel, but $\alpha < \phi_i < \phi_j$) 
    - [x] edit plotting in the paper and change wording (how $\eta$ is estimated and quote `mev` package)
  - [x] Try a $N = 1,000,000,000$ version?
    - Not enough memory on misspiggy, so we can't go one additional magnitude larger
    - Running a $N = 500,000,000$ version. `~420G`
    - RAM is still an issue
      - break down generating data and calculating `qRW` in separate pieces
      - <mark>$\eta_{34}$ still not good</mark>
      - <mark> e.g. $\eta_{45}$ estimation doesn't "monotonically" gets better with larger $N$</mark>
- [x] Use Likun's code to find the limit of $\eta$ (at least get the limit right):
  - implemented and tried $N = 1,000,000$ but sample size might still be small?
    - Check: <mark>scale or shape parameter fit?</mark>
  - Can't add order of magnitude on $N$ for `mev`, currently running a $N = 100,000,000$ to estimate limit $\eta$. Results are copied in `limit_eta.R` file.
    - Also trying to match $N$ to 300,000,000?
      - should not try while `python mev` is running, they fight for RAM.
      - [x] waiting until `mev` finishes to try this.
  - `gPdtest` is not available anymore, using `extRemes` instead
    - `qevd` function in `extRemes` <mark> seems wrong</mark>, as it gives negative values when marginally transform to Frechet. 
    - Using a manual implementation to transform to Frechet; or use the `qfrechet` function in `VGAM`. They seem to give equal answers.
    - Check manual implementation that for  <mark> unit Frechet, $\alpha = 1$. </mark>
  - just put a point at the end and connect with the previous scatter point, also try to save the data (point (x,y)) so re-plotting is easier

- [x] Fix up MCMC details Appendix section B
  - Do we still need this appendix? almost all things are already in section 3
  - shovel a lot of things into this appendix and shorten the paper?

- <mark>How do we want to justify $\gamma_k = 0.5$?</mark>

- Figure placement seems annoying to modify. Do this lastly.

## Oct. 1 Tuesday Meeting with Mark/Ben

### Meeting

- Talked about the difficulties in estimating $\eta$, evening using `mev` package
- Go over the comments in the paper
  - Ben will look at the "unresolved" comments

### Todo

#### Paper

- [x] Talk about the difficulties of $\eta$ not reaching the empirical bounds
  ```
  eta = np.log(1-us) / np.log(prob_co_extreme)
  ```
  $\eta = \dfrac{\log(1-u)}{\log(P(F_1(X_1) > u, F_2(X_2) > u))}$ -- the empirical estimates on the denominator is troubling.
- [x] Move model integral into section 3 together with the likelihood
- [x] Follow Ben's edit:
  - [x] Section 3
- [x] Change the data analysis section 5's intro paragraphs
  - if any issue, keep some notes for Mark
- [x] Update model names in the section 5's texts
- [x] Elaborate bullet points in section 6's discussion into paragraphs

#### simulation
- [x] try hill estimator from `mev` if time permits
  - `mev`'s Hill estimator is using MLE for a scale parameter(?) (of an exponential??)
  - our margin is not in Exponential, so transform to uniform and then transform to Exponential margin. 
  - `hill` did great for AD cases, poorly for AI cases as always.


## Sept. 24 Tuesday Meeting with Likun/Ben

### Meeting

Went over changes made on plots: 
  - Figure 3 the Thm2.3 simulation plot
  - Current stage of the Figure 4 (empirical estimates of $\chi$ and $\eta$)
  - the simulation scenarios plots 3D -> 2D
  - the data application posterior single hue plots
  - The moving window $\chi$-plots are good as they are now.

### Todo

### Poster

- [x] Ben's "p-card" -- purchasing card?

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

- [x] Follow Ben's edits
- [x] Through out the paper, change "marginal" to "univariate"
- [x] Through out the paper, change "copula" to "dependence model"
- [-] Update model names in the manuscript texts


### Simulation $\chi$ and $\eta$ - Example 3 Figure 4

- Computation:
  - [x] save and load data
  - [x] fix the bug in $\eta$ estimation and redo estimations
  - [x] eventually use $N = 300,000,000$ datapoints
  - [x] Using `mev` in `R` to estimate $\eta$ because treating $L(1-u)$ as 1 may be biased
    - Seems it also ignores $L(1-u)$ from its vignette
    - Gives same if not worse result than manually estimate empirically
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
  - if $\chi$ is not smooth, use smoothing splines to smooth it



## Sept. 17 Tuesday Meeting with Likun/Mark/Ben

### Paper - Ben will give a pass on Thursday

*Do this first, Ben will give a pass on Thursday*

- Section 2 Model
  - [x] Change the "marginal distribution of the copula" to univariate distribution of the dependence model -- **through out the paper!**
  - [x] Remade Figure 3 ($\phi(s)$ surface) Figure 4 (empirical $\chi$ and $\eta$) so that we are not making radius able to change and have plots verifying the bounds in Thm 2.3
    - [x] Figure 3 - simulation case
    - [x] Figure 4 - empirical plots
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
  - [-] Update model names in the manuscript texts

### Re-do $\eta$ and $\chi$ empricial estimation for Thm 2.3

- [x] Empirically estimate $\chi$ and $\eta$
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
