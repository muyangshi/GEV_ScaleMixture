"""
Proposal Covariance Matrix used to initialize the Sigma_0
Only used when starting the chain fresh, 
as the laster daisychain will load the proposal scalar variance and cov from pickle files
"""

import numpy as np

sigma_Beta_mu0_cov      = None
sigma_Beta_mu1_cov      = None
sigma_Beta_logsigma_cov = None
sigma_Beta_ksi_cov      = None
phi_cov                 = None
range_cov               = None
Beta_mu0_cov            = None
Beta_mu1_cov            = None
Beta_logsigma_cov       = None
Beta_ksi_cov            = None
R_log_cov               = None

sigma_Beta_mu0_cov      = 2.24219783
sigma_Beta_mu1_cov      = 0.04592726
sigma_Beta_logsigma_cov = 0.56274039
sigma_Beta_ksi_cov      = 0.17541573
phi_cov                 = np.array([[ 1.71127398e-02, -7.60894815e-04,  1.44457423e-03,
        -2.84663642e-04,  3.80906921e-04, -2.37198922e-04,
         1.83873549e-03, -1.92170548e-04,  1.81312348e-04,
        -7.59587972e-04,  3.04839794e-04, -5.58281389e-04,
        -1.08005190e-04],
       [-7.60894815e-04,  1.55561491e-02, -9.14240396e-05,
         1.30306361e-03,  1.00910853e-04,  7.07414500e-04,
        -3.93545969e-04, -1.97100716e-04, -8.03570116e-04,
        -4.81073002e-04, -1.74363810e-03,  2.92494913e-05,
         1.09617601e-03],
       [ 1.44457423e-03, -9.14240396e-05,  1.35002494e-02,
        -3.40062506e-04,  6.57553248e-04,  9.47790208e-04,
        -6.83424547e-04,  1.30436344e-04, -2.26111715e-04,
         1.80100753e-03, -8.54398972e-04,  2.49541091e-04,
         2.65227168e-04],
       [-2.84663642e-04,  1.30306361e-03, -3.40062506e-04,
         1.18522318e-02,  4.31379102e-04,  1.08735042e-03,
        -7.37241957e-04, -5.85525750e-04, -6.08740354e-04,
        -7.93092704e-04, -4.92033528e-06, -1.43294877e-03,
         1.08677109e-03],
       [ 3.80906921e-04,  1.00910853e-04,  6.57553248e-04,
         4.31379102e-04,  1.47232195e-02, -4.34441416e-04,
         2.90067795e-04, -3.76309614e-04,  1.91836211e-03,
         1.29039100e-03,  2.66930983e-04,  5.00971425e-04,
         2.82528206e-04],
       [-2.37198922e-04,  7.07414500e-04,  9.47790208e-04,
         1.08735042e-03, -4.34441416e-04,  1.43331380e-02,
         6.60362236e-04,  4.39761450e-04, -9.94482106e-04,
         5.53074069e-04, -4.77826300e-04, -8.33754271e-04,
        -2.83779431e-04],
       [ 1.83873549e-03, -3.93545969e-04, -6.83424547e-04,
        -7.37241957e-04,  2.90067795e-04,  6.60362236e-04,
         2.21176891e-02, -5.96682306e-05, -6.90945077e-04,
         5.00985377e-04,  1.54283116e-03, -1.40435619e-03,
        -6.21317966e-04],
       [-1.92170548e-04, -1.97100716e-04,  1.30436344e-04,
        -5.85525750e-04, -3.76309614e-04,  4.39761450e-04,
        -5.96682306e-05,  1.05536374e-02,  2.02553076e-04,
        -4.76256488e-04,  1.08588670e-04, -1.65582181e-04,
        -2.99237546e-04],
       [ 1.81312348e-04, -8.03570116e-04, -2.26111715e-04,
        -6.08740354e-04,  1.91836211e-03, -9.94482106e-04,
        -6.90945077e-04,  2.02553076e-04,  1.25325497e-02,
        -1.09055280e-03,  9.31705797e-04,  6.71528367e-04,
        -8.49843521e-04],
       [-7.59587972e-04, -4.81073002e-04,  1.80100753e-03,
        -7.93092704e-04,  1.29039100e-03,  5.53074069e-04,
         5.00985377e-04, -4.76256488e-04, -1.09055280e-03,
         1.71681644e-02,  2.13898515e-04,  3.77068848e-04,
         1.03138313e-03],
       [ 3.04839794e-04, -1.74363810e-03, -8.54398972e-04,
        -4.92033528e-06,  2.66930983e-04, -4.77826300e-04,
         1.54283116e-03,  1.08588670e-04,  9.31705797e-04,
         2.13898515e-04,  1.33858145e-02,  6.93292291e-04,
         9.10014340e-04],
       [-5.58281389e-04,  2.92494913e-05,  2.49541091e-04,
        -1.43294877e-03,  5.00971425e-04, -8.33754271e-04,
        -1.40435619e-03, -1.65582181e-04,  6.71528367e-04,
         3.77068848e-04,  6.93292291e-04,  1.18438340e-02,
         1.43448328e-04],
       [-1.08005190e-04,  1.09617601e-03,  2.65227168e-04,
         1.08677109e-03,  2.82528206e-04, -2.83779431e-04,
        -6.21317966e-04, -2.99237546e-04, -8.49843521e-04,
         1.03138313e-03,  9.10014340e-04,  1.43448328e-04,
         1.36516210e-02]])
range_cov               = np.array([[ 1.31336733e+00,  1.20308878e-01,  2.04317136e-01,
        -9.33045539e-02,  7.64721890e-02, -1.97645171e-01,
         4.35718914e-02,  3.62509497e-02,  4.26521587e-02,
         1.98417777e-01, -5.90621037e-02, -2.43882123e-02,
         3.02143661e-02],
       [ 1.20308878e-01,  1.73093106e+00,  1.64427864e-01,
        -9.18307780e-02,  6.34591992e-02, -8.99877550e-02,
         7.04752697e-02,  2.85136324e-01, -1.79743643e-02,
        -1.95816581e-02, -6.31256610e-02, -3.11900462e-02,
         4.25248231e-03],
       [ 2.04317136e-01,  1.64427864e-01,  1.31616177e+00,
        -6.81801718e-02,  5.97777516e-02, -2.44719228e-01,
        -1.22117691e-01,  5.18259914e-03, -3.28538614e-02,
         2.08344212e-01,  3.74565560e-02, -5.02612612e-03,
        -4.58512149e-02],
       [-9.33045539e-02, -9.18307780e-02, -6.81801718e-02,
         1.45383705e+00, -3.60635662e-02,  1.60663585e-02,
        -5.12980705e-02,  7.88542965e-05,  3.64648611e-02,
         9.32380638e-02,  4.69131307e-02, -4.33148744e-03,
         4.89433939e-02],
       [ 7.64721890e-02,  6.34591992e-02,  5.97777516e-02,
        -3.60635662e-02,  1.30362096e+00,  1.89948594e-01,
         7.95616600e-02, -5.60277045e-02, -2.76772811e-02,
        -1.50795949e-03,  6.78289177e-02, -6.59938662e-02,
         3.94696780e-02],
       [-1.97645171e-01, -8.99877550e-02, -2.44719228e-01,
         1.60663585e-02,  1.89948594e-01,  2.04702038e+00,
         4.10112077e-01, -1.79228080e-01, -1.13810548e-02,
        -1.87835831e-01, -7.70304017e-02, -6.94810105e-02,
        -1.15397392e-01],
       [ 4.35718914e-02,  7.04752697e-02, -1.22117691e-01,
        -5.12980705e-02,  7.95616600e-02,  4.10112077e-01,
         2.55345454e+00,  7.17746147e-02, -1.20127577e-02,
         3.82621168e-02, -9.01431919e-02, -3.84752958e-02,
        -1.02342004e-01],
       [ 3.62509497e-02,  2.85136324e-01,  5.18259914e-03,
         7.88542965e-05, -5.60277045e-02, -1.79228080e-01,
         7.17746147e-02,  2.12041043e+00,  9.22324055e-03,
         1.36743572e-01, -1.83587865e-01,  1.81448504e-01,
        -1.34796264e-01],
       [ 4.26521587e-02, -1.79743643e-02, -3.28538614e-02,
         3.64648611e-02, -2.76772811e-02, -1.13810548e-02,
        -1.20127577e-02,  9.22324055e-03,  8.74822340e-02,
        -1.33302441e-01, -8.78521410e-02, -1.09079192e-01,
         3.72429403e-02],
       [ 1.98417777e-01, -1.95816581e-02,  2.08344212e-01,
         9.32380638e-02, -1.50795949e-03, -1.87835831e-01,
         3.82621168e-02,  1.36743572e-01, -1.33302441e-01,
         2.34758103e+00,  4.52577247e-01,  1.20337846e+00,
        -1.31270526e-01],
       [-5.90621037e-02, -6.31256610e-02,  3.74565560e-02,
         4.69131307e-02,  6.78289177e-02, -7.70304017e-02,
        -9.01431919e-02, -1.83587865e-01, -8.78521410e-02,
         4.52577247e-01,  6.23002632e-01,  5.06582400e-01,
         2.19287338e-02],
       [-2.43882123e-02, -3.11900462e-02, -5.02612612e-03,
        -4.33148744e-03, -6.59938662e-02, -6.94810105e-02,
        -3.84752958e-02,  1.81448504e-01, -1.09079192e-01,
         1.20337846e+00,  5.06582400e-01,  1.21566854e+00,
        -2.05390158e-01],
       [ 3.02143661e-02,  4.25248231e-03, -4.58512149e-02,
         4.89433939e-02,  3.94696780e-02, -1.15397392e-01,
        -1.02342004e-01, -1.34796264e-01,  3.72429403e-02,
        -1.31270526e-01,  2.19287338e-02, -2.05390158e-01,
         2.00324728e-01]])
Beta_mu0_cov            = np.array([[ 3.15321346e+01, -1.63125996e+00, -2.74406104e+01,
        -1.46866868e-01, -1.71120046e-01,  2.12965882e+00,
        -9.00196363e-01,  2.18544681e-01, -2.93162778e+00,
        -7.25663307e-02, -5.02974840e-01, -1.35102877e+00,
         1.97501575e+00],
       [-1.63125996e+00,  1.41666821e+00, -5.95453083e-01,
         2.08632071e-01, -5.27633945e-01,  3.40938974e-01,
         3.73087960e-01, -1.44363273e-01,  9.07044374e-01,
        -2.39754412e-01,  1.66500773e-01,  1.34865547e+00,
        -1.12868072e-01],
       [-2.74406104e+01, -5.95453083e-01,  3.33133534e+01,
         4.54295781e-01,  1.21510230e+00,  2.40761073e-01,
         1.08283965e+00,  1.81324564e-01,  4.23552620e+00,
         8.03291802e-01,  3.43684657e-01, -8.81094359e-01,
         8.22871143e-02],
       [-1.46866868e-01,  2.08632071e-01,  4.54295781e-01,
         7.26797757e-01, -3.84992208e-02,  4.65804132e-01,
         1.08758258e-01,  4.48158242e-02,  2.54575000e-01,
        -1.41345563e-01, -5.42766843e-02,  3.05815447e-01,
         1.97491966e-01],
       [-1.71120046e-01, -5.27633945e-01,  1.21510230e+00,
        -3.84992208e-02,  1.09692415e+00, -1.77987917e-01,
         1.26767987e-01,  4.02832465e-02, -5.24841754e-01,
         2.03846780e-02, -1.87003519e-02, -6.69313065e-01,
         6.05913606e-02],
       [ 2.12965882e+00,  3.40938974e-01,  2.40761073e-01,
         4.65804132e-01, -1.77987917e-01,  4.22823954e+00,
         5.84646032e-01,  3.50470775e-02,  5.44146686e-01,
        -5.18755146e-02,  8.81329933e-03,  2.87603796e-01,
         2.87971142e+00],
       [-9.00196363e-01,  3.73087960e-01,  1.08283965e+00,
         1.08758258e-01,  1.26767987e-01,  5.84646032e-01,
         1.79550276e+00, -1.82503609e-01,  3.29023762e-01,
        -9.68521932e-04,  1.36729150e-01, -6.03941518e-01,
         6.11408374e-01],
       [ 2.18544681e-01, -1.44363273e-01,  1.81324564e-01,
         4.48158242e-02,  4.02832465e-02,  3.50470775e-02,
        -1.82503609e-01,  4.05671721e-01, -4.93173135e-02,
        -3.50835199e-02, -8.54230959e-02,  3.34037080e-02,
         5.42782998e-02],
       [-2.93162778e+00,  9.07044374e-01,  4.23552620e+00,
         2.54575000e-01, -5.24841754e-01,  5.44146686e-01,
         3.29023762e-01, -4.93173135e-02,  3.12277881e+00,
        -8.85085514e-03,  1.22678674e-01,  1.24160524e+00,
        -1.89298902e-01],
       [-7.25663307e-02, -2.39754412e-01,  8.03291802e-01,
        -1.41345563e-01,  2.03846780e-02, -5.18755146e-02,
        -9.68521932e-04, -3.50835199e-02, -8.85085514e-03,
         1.64397595e+00, -5.99483322e-02, -1.45761443e+00,
         1.00724183e-01],
       [-5.02974840e-01,  1.66500773e-01,  3.43684657e-01,
        -5.42766843e-02, -1.87003519e-02,  8.81329933e-03,
         1.36729150e-01, -8.54230959e-02,  1.22678674e-01,
        -5.99483322e-02,  3.82710605e-01,  1.58456143e-01,
         2.09769456e-02],
       [-1.35102877e+00,  1.34865547e+00, -8.81094359e-01,
         3.05815447e-01, -6.69313065e-01,  2.87603796e-01,
        -6.03941518e-01,  3.34037080e-02,  1.24160524e+00,
        -1.45761443e+00,  1.58456143e-01,  3.81082091e+00,
        -6.57765097e-01],
       [ 1.97501575e+00, -1.12868072e-01,  8.22871143e-02,
         1.97491966e-01,  6.05913606e-02,  2.87971142e+00,
         6.11408374e-01,  5.42782998e-02, -1.89298902e-01,
         1.00724183e-01,  2.09769456e-02, -6.57765097e-01,
         2.94806047e+00]])
Beta_mu1_cov            = np.array([[ 0.23216379, -0.05263032, -0.02491087, -0.01678562, -0.00381033,
         0.04092088, -0.02972265,  0.0093179 ,  0.05209236, -0.02182003,
        -0.00347718, -0.00857976,  0.00352992],
       [-0.05263032,  0.08443322, -0.05954497,  0.00488946, -0.0199205 ,
         0.02017577,  0.02713258, -0.00439456,  0.0309929 ,  0.02097142,
        -0.00215148,  0.03708755, -0.00307177],
       [-0.02491087, -0.05954497,  0.24362702, -0.00745658,  0.01813185,
         0.00367836, -0.01663199,  0.01420457,  0.03144024, -0.00422348,
         0.01517999, -0.03961581, -0.00638841],
       [-0.01678562,  0.00488946, -0.00745658,  0.15818594,  0.02687011,
        -0.02278701,  0.03032663, -0.01719754, -0.03302199, -0.00266835,
        -0.00453857,  0.01561272,  0.02869446],
       [-0.00381033, -0.0199205 ,  0.01813185,  0.02687011,  0.14942064,
        -0.03306617,  0.03125897, -0.01066983, -0.05414886, -0.01700102,
        -0.00209744, -0.00132846,  0.01723597],
       [ 0.04092088,  0.02017577,  0.00367836, -0.02278701, -0.03306617,
         0.21089446, -0.01548101,  0.01345348,  0.02278242,  0.01344348,
        -0.0109312 ,  0.01472095,  0.05308414],
       [-0.02972265,  0.02713258, -0.01663199,  0.03032663,  0.03125897,
        -0.01548101,  0.17290101, -0.01114417, -0.03231672, -0.03819242,
         0.00104469, -0.02661493,  0.03259321],
       [ 0.0093179 , -0.00439456,  0.01420457, -0.01719754, -0.01066983,
         0.01345348, -0.01114417,  0.10279111,  0.01470863, -0.00095727,
        -0.01155957,  0.00863621, -0.00885678],
       [ 0.05209236,  0.0309929 ,  0.03144024, -0.03302199, -0.05414886,
         0.02278242, -0.03231672,  0.01470863,  0.21346281,  0.00885201,
        -0.00416265,  0.03151548, -0.02826532],
       [-0.02182003,  0.02097142, -0.00422348, -0.00266835, -0.01700102,
         0.01344348, -0.03819242, -0.00095727,  0.00885201,  0.20063942,
        -0.00623949, -0.05537094, -0.00306649],
       [-0.00347718, -0.00215148,  0.01517999, -0.00453857, -0.00209744,
        -0.0109312 ,  0.00104469, -0.01155957, -0.00416265, -0.00623949,
         0.12062285,  0.0019279 ,  0.00085917],
       [-0.00857976,  0.03708755, -0.03961581,  0.01561272, -0.00132846,
         0.01472095, -0.02661493,  0.00863621,  0.03151548, -0.05537094,
         0.0019279 ,  0.17641369, -0.00326842],
       [ 0.00352992, -0.00307177, -0.00638841,  0.02869446,  0.01723597,
         0.05308414,  0.03259321, -0.00885678, -0.02826532, -0.00306649,
         0.00085917, -0.00326842,  0.19402043]])
Beta_logsigma_cov       = np.array([[ 0.00452845, -0.00120254],
       [-0.00120254,  0.00061841]])
Beta_ksi_cov            = np.array([[ 0.00252992, -0.00071647],
       [-0.00071647,  0.0003614 ]])
