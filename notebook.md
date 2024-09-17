# Progress on GEV Project

## Sept 17 Tuesday Meeting with Likun/Mark/Ben

### Paper - Ben will give a pass on Thursday

*Do this first, Ben will give a pass on Thursday*

- Section 2 Model
  - [ ] Change the "marginal distribution of the copula" to univariate distribution of the dependence model -- **through out the paper!**
  - [ ] Remade Figure 3 ($\phi(s)$ surface) Figure 4 (empirical $\chi$ and $\eta$) so that we are not making radius able to change and have plots verifying the bounds in Thm 2.3
- Section 3 Bayesian Inference
  - [ ] add hierarchical model into section 3
  - [ ] paragraph on knot (spatial dimension reduction) for the dependence model (don't say process model because levels are not clear)
  - [ ] move the splines about marginal data model to section 5
- Section 4 Simulation Study
  - [ ] Figure 3 has already used scenario. Change simulation scenario to case. (Maybe unneccessary as we will redo Figure 3 with just one scenario)
- Section 5 Data Application
  - [ ] $\chi$-plot comment who matches with who; model realization has gradually fading color indicating AI (which is good?)

### Re-do $\eta$ and $\chi$ empricial estimation for Thm 2.3

- [ ] Empirically estimate $\chi$ and $\eta$
  - Treat $L_Z(...)$ as 1 since we are approaching the limit. Likun: do a regression on (log of) the formula to get $\eta$
- $\eta$ fits in the story-telling because there are bounds in the Theorem (red dashed lines), and empirical result can show "they are good bounds"
- **Same** simulation with special placements of the points should be sufficient; we also don't want to change the radii. 

### Plotting

- [ ] Change the $\phi(s)$ surface plot so that 0.5 is white
- [ ] Change the $\rho(s)$ surface plot to single color
- [ ] Potentially also changing the $\theta_{GEV}(s)$ plots to have one hue?
- [ ] Change the bound of colorbar for $\chi$-plot to [0, 0.5].



