"""
20250529
    - use qRW_NN to emulate qRW

reasoning for choosing the bounds:
    - gamma_bar: we chose gamma_k as 0.5, gamma_bar won't be smaller than this

    # exponentially increase the number of points of p towards 1
    - p = p_min + (p_max - p_min) * (1 - np.exp(-BETA * u)) / (1 - np.exp(-BETA))

Notes on coding:
    - pRW(1e16, 1, 4, 50) yields array(0.99999999)

    - This is incredibly slow -- much better to directly pass X_2d to model.predict
        def qRW_NN(x, phi, gamma, tau):
            return model.predict(np.array([[x, phi, gamma, tau]]), verbose=0)[0]
        qRW_NN_vec = np.vectorize(qRW_NN)

    - 400,000 qRW() evals in 5 minutes, 30 processes

    - Windows/Mac need to explicitly use 'fork':
        from multiprocessing import get_context
        p = get_context("fork").Pool(4)
        results = p.map(pRW_par, inputs)
        p.close()
"""

# %% imports

# base python
import multiprocessing
import os
import time
import datetime
import pickle
from tqdm import tqdm

# packages
import scipy
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf
from   tensorflow        import keras
from   scipy.stats       import qmc

# custom modules
from utilities           import *
print('link function:', norm_pareto, 'Pareto')

# Training Settings -----------------------------------------------------------

# system settings -----------------------------------------
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)
keras.mixed_precision.set_global_policy('mixed_float16')
GLOBAL_SEED = 531
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
N_CORES  = 7 if multiprocessing.cpu_count() < 64 else 64

# script settings -----------------------------------------
GENERATE = False # Generate design points and true qRW values
TRAIN    = False # Train qRW_NN on generated data
SIMULATE = True  # simulate data for marginal likelihood surface

# GENERATE settings ---------------------------------------
N     = int(1e8)
N_val = int(1e5)
d     = 3 # p, phi, gamma
ALPHA                = 1.0  # coefficient for weighted MSE
BETA                 = 1.0  # coefficient for exponential increase sampling towards 1 for p = p_min + (p_max - p_min) * (1 - np.exp(-BETA * u)) / (1 - np.exp(-BETA))
p_min,     p_max     = 0.1, 0.99999
phi_min,   phi_max   = 0.05, 0.95
gamma_min, gamma_max = 0.5,  5

# TRAIN settings ------------------------------------------
INITIAL_EPOCH         = 0
N_EPOCHS              = 200
BATCH_SIZE            = 4096
# STEPS_PER_EPOCH       = N // BATCH_SIZE
VALIDATION_BATCH_SIZE = 4096
N_EPOCH_PER_DECAY     = 1
UNIT_HYPERCUBE        = True
LOG_RESPONSE          = False
EVGAN_RESPONSE        = True

# Helper Functions ------------------------------------------------------------

# @keras.utils.register_keras_serializable(package="Custom")
def weighted_mse(alpha=1.0, eps=1e-8):
    def loss_fn(y_true, y_pred):
        # define the weights
        weights = 1.0 + alpha * tf.math.softplus(y_true)

        # weighted sum of squared errors
        se      = keras.backend.square(y_pred - y_true)
        sse     = keras.backend.sum(weights * se)

        # normalize by the sum of weights
        wmse    = sse / (keras.backend.sum(weights) + eps)

        return wmse

    return loss_fn

def qRW_par(args): # wrapper to put qRW for multiprocessing
    p, phi, gamma = args
    return(qRW(p, phi, gamma))

def H(y, p):
    return -np.log(y) / (np.log(1-p**2) - np.log(2))

def H_inv(h, p):
    return np.exp(-h * (np.log(1-p**2) - np.log(2)))


# %% LHS design for the parameter of qRW(p, phi, gamma)

if GENERATE:
    lhs_sampler = qmc.LatinHypercube(d, scramble = False, seed = GLOBAL_SEED)
    lhs_samples = lhs_sampler.random(N) # doesn't include the boundary
    lhs_samples = np.vstack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary

    # linear scaling for phi, gamma;
    # exponentially scale p towards 1
    l_bounds   = [0, phi_min, gamma_min]
    u_bounds   = [1, phi_max, gamma_max]
    X_lhs      = qmc.scale(lhs_samples, l_bounds, u_bounds)
    X_lhs[:,0] = p_min + (p_max - p_min) * (1 - np.exp(-BETA * X_lhs[:,0]))/(1-np.exp(-BETA))

    # Calculate the design points

    with multiprocessing.get_context('fork').Pool(processes=N_CORES) as pool:
        Y_lhs = list(tqdm(pool.imap(qRW_par, list(X_lhs)), total=len(X_lhs), desc='qRW'))
    Y_lhs = np.array(Y_lhs)

    np.save(rf'qRW_X_{N}.npy', X_lhs)
    np.save(rf'qRW_Y_{N}.npy', Y_lhs)

    # Caluclate a set of validation points

    lhs_sampler_val = qmc.LatinHypercube(d, scramble = False, seed = GLOBAL_SEED+1)
    lhs_samples_val = lhs_sampler_val.random(N_val) # doesn't include the boundary
    lhs_samples_val = np.row_stack(([0]*d, lhs_samples_val, [1]*d)) # manually add the boundary
    l_bounds        = [0, phi_min, gamma_min]
    u_bounds        = [1, phi_max, gamma_max]
    X_lhs_val       = qmc.scale(lhs_samples_val, l_bounds, u_bounds)
    X_lhs_val[:,0]  = p_min + (p_max - p_min) * (1 - np.exp(-BETA * X_lhs_val[:,0]))/(1-np.exp(-BETA))

    with multiprocessing.get_context('fork').Pool(processes=N_CORES) as pool:
        Y_lhs_val = list(tqdm(pool.imap(qRW_par, list(X_lhs_val)), total=len(X_lhs_val), desc='qRW_val'))
    Y_lhs_val = np.array(Y_lhs_val)

    np.save(rf'qRW_X_val_{N_val}.npy', X_lhs_val)
    np.save(rf'qRW_Y_val_{N_val}.npy', Y_lhs_val)

# %% load design points and train
if TRAIN:
    X_train = np.load(rf'qRW_X_{N}.npy')
    y_train = np.load(rf'qRW_Y_{N}.npy')
    X_val   = np.load(rf'qRW_X_val_{N_val}.npy')
    y_val   = np.load(rf'qRW_Y_val_{N_val}.npy')

    if LOG_RESPONSE:
        print('taking log of response...')
        y_train = np.log(y_train)
        y_val   = np.log(y_val)

    if EVGAN_RESPONSE:
        print('taking EVGAN response...')
        y_train = H(y_train, X_train[:,0])
        y_val   = H(y_val, X_val[:,0])

    if UNIT_HYPERCUBE:
        X_min   = np.min(X_train, axis = 0)
        X_max   = np.max(X_train, axis = 0)
        print('X_min:', X_min)
        print('X_max:', X_max)

        print('scaling X into unit hypercube...')
        X_train = (X_train - X_min)/(X_max - X_min)
        X_val   = (X_val - X_min)/(X_max - X_min)

        np.save('qRW_NN_X_min.npy', X_min)
        np.save('qRW_NN_X_max.npy', X_max)

    # Defining model ----------------------------------------------------------

    if INITIAL_EPOCH == 0:
        model = keras.Sequential(
            [
                keras.Input(shape=(d,)),
                keras.layers.Dense(512, activation = 'tanh'),
                keras.layers.Dense(512, activation = 'tanh'),
                keras.layers.Dense(512, activation = 'tanh'),
                keras.layers.Dense(1,   activation = 'linear')
        ])
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-3,
            decay_steps           = N_EPOCH_PER_DECAY * len(y_train) // BATCH_SIZE,
            decay_rate            = 0.96,
            staircase             = False
        )
    else:
        model = keras.models.load_model('./checkpoint.model.keras')
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-4,
            decay_steps           = N_EPOCH_PER_DECAY * len(y_train) // BATCH_SIZE,
            decay_rate            = 0.96,
            staircase             = False
        )

    if LOG_RESPONSE:   loss_func = weighted_mse(alpha=ALPHA)
    if EVGAN_RESPONSE: loss_func = keras.losses.MeanSquaredError()

    model.compile(
        optimizer   = keras.optimizers.Adam(learning_rate=lr_schedule),
        # loss        = keras.losses.MeanSquaredError(),
        # loss        = weighted_mse(alpha=ALPHA),
        loss        = loss_func,
        jit_compile = True)
    model.summary()


    # Fitting Model -----------------------------------------------------------

    class LossHistoryPlotter(keras.callbacks.Callback):
        def __init__(self, plot_every = 10, output_dir='./'):
            super().__init__()
            self.plot_every = plot_every
            self.output_dir = output_dir
            self.epoch_loss = []
            self.epoch_val_loss = []

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_loss.append(logs['loss'])
            self.epoch_val_loss.append(logs['val_loss'])
            if (epoch+1) % self.plot_every == 0:
                plt.figure()
                plt.plot(range(1, len(self.epoch_loss)+1), self.epoch_loss, label='Training Loss')
                plt.plot(range(1, len(self.epoch_val_loss)+1), self.epoch_val_loss, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss Curves up to Epoch {epoch}')
                plt.xlim(INITIAL_EPOCH, INITIAL_EPOCH + N_EPOCHS)
                plt.legend()
                plt.savefig(f'Plot_loss_during_train.pdf')
                plt.close()
    loss_plotter = LossHistoryPlotter(plot_every = 10)

    checkpoint_filepath = './checkpoint.model.keras' # only saves the best performer seen so far after each epoch
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True)

    start_time = time.time()
    print('started fitting NN:', datetime.datetime.now())

    history = model.fit(
        X_train,
        y_train,
        initial_epoch         = INITIAL_EPOCH,
        epochs                = N_EPOCHS,
        batch_size            = BATCH_SIZE,
        validation_batch_size = VALIDATION_BATCH_SIZE,
        verbose = 2,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint_callback, loss_plotter])

    end_time = time.time()
    print('done:', round(end_time - start_time, 3))

    with open(rf'trainHistoryDict_{INITIAL_EPOCH}to{N_EPOCHS}.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    if LOG_RESPONSE: bestmodel = keras.models.load_model(checkpoint_filepath,
                                                         custom_objects={'loss_fn': weighted_mse(alpha=ALPHA)})
    if EVGAN_RESPONSE: bestmodel = keras.models.load_model(checkpoint_filepath)
    bestmodel.save(rf'./qRW_NN_{N}.keras')
    with open(rf'qRW_NN_{N}_weights_and_biases.pkl', 'wb') as file:
        pickle.dump(bestmodel.get_weights(), file, protocol=pickle.HIGHEST_PROTOCOL)

    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('validation loss')
    plt.savefig(rf'Plot_val_loss_{INITIAL_EPOCH}to{N_EPOCHS}.pdf')
    plt.show()
    plt.close()

    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('training loss')
    plt.savefig(rf'Plot_train_loss_{INITIAL_EPOCH}to{N_EPOCHS}.pdf')
    plt.show()
    plt.close()

    # %% Prediction --------------------------------------------------------------

    # load model weights
    with open(rf'qRW_NN_{N}_weights_and_biases.pkl', 'rb') as f:
        w0, b0, w1, b1, w2, b2, w3, b3 = pickle.load(f)
    X_min    = np.load('qRW_NN_X_min.npy')
    X_max    = np.load('qRW_NN_X_max.npy')

    def NN_forward_pass(X_scaled):
        # Layer 1
        Z = X_scaled @ w0 + b0
        Z = np.tanh(Z)
        # Layer 2
        Z = Z @ w1 + b1
        Z = np.tanh(Z)
        # Layer 3
        Z = Z @ w2 + b2
        Z = np.tanh(Z)
        # Layer 4
        Z = Z @ w3 + b3
        return Z

    def qRW_NN(p_vec, phi_vec, gamma_vec): # p_vec, phi_vec, gamma_vec are on original scale
        # check input shape and broadcast
        p_vec     = np.atleast_1d(p_vec)
        phi_vec   = np.atleast_1d(phi_vec)
        gamma_vec = np.atleast_1d(gamma_vec)
        if not (len(p_vec) == len(phi_vec) == len(gamma_vec)): max_length = np.max([len(p_vec), len(phi_vec), len(gamma_vec)])
        if len(p_vec)      == 1: p_vec     = np.full(max_length, p_vec[0])
        if len(phi_vec)    == 1: phi_vec   = np.full(max_length, phi_vec[0])
        if len(gamma_vec)  == 1: gamma_vec = np.full(max_length, gamma_vec[0])
        if not (len(p_vec) == len(phi_vec) == len(gamma_vec)): raise ValueError('Cannot broadcast with different lengths.')

        # make prediction
        X        = np.column_stack((p_vec, phi_vec, gamma_vec))
        X_scaled = qmc.scale(X, X_min, X_max, reverse = True)

        if LOG_RESPONSE:
            return np.exp(NN_forward_pass(X_scaled))
        if EVGAN_RESPONSE:
            Z = NN_forward_pass(X_scaled).ravel()
            return H_inv(Z, p_vec)

    def qRW_NN_2p(p_vec, phi_vec, gamma_vec):
        # check input shape and broadcast
        p_vec     = np.atleast_1d(p_vec)
        phi_vec   = np.atleast_1d(phi_vec)
        gamma_vec = np.atleast_1d(gamma_vec)
        if not (len(p_vec) == len(phi_vec) == len(gamma_vec)): max_length = np.max([len(p_vec), len(phi_vec), len(gamma_vec)])
        if len(p_vec)      == 1: p_vec     = np.full(max_length, p_vec[0])
        if len(phi_vec)    == 1: phi_vec   = np.full(max_length, phi_vec[0])
        if len(gamma_vec)  == 1: gamma_vec = np.full(max_length, gamma_vec[0])
        if not (len(p_vec) == len(phi_vec) == len(gamma_vec)): raise ValueError('Cannot broadcast with different lengths.')

        # check proportion within interpolation range
        condition_p     = (0.1  <= p_vec)     & (p_vec     <= 0.9999)
        condition_phi   = (0.05 <= phi_vec)   & (phi_vec   <= 0.95)
        condition_gamma = (0.5  <= gamma_vec) & (gamma_vec <= 5)
        condition       = condition_p & condition_phi & condition_gamma
        if np.mean(condition) < 0.9:
            print(f'Warning: {np.mean(condition)} of the inputs are inside the range of the NN.')
            print('Proportion p interpolated:',     np.mean(condition_p))
            print('Proportion phi interpolated:',   np.mean(condition_phi))
            print('Proportion gamma interpolated:', np.mean(condition_gamma))

        # make prediction
        outputs = np.full((len(p_vec),), fill_value=np.nan)
        outputs[condition] = qRW_NN(p_vec[condition], phi_vec[condition], gamma_vec[condition]).ravel()
        outputs[~condition] = qRW(p_vec[~condition], phi_vec[~condition], gamma_vec[~condition]).ravel()
        return outputs

    # %% Evaluation --------------------------------------------------------------

    # Prediction Plots ------------------------------------

    phi      = 0.5
    gamma    = 0.5
    ps       = np.linspace(0.1, 0.9999, 100)
    qRW_true = qRW(ps, phi, gamma)
    qRW_pred = qRW_NN(ps, phi, gamma)

    plt.plot(ps, qRW_true, 'k.-', label = 'truth')
    plt.plot(ps, qRW_pred, 'b.-', label = 'emulated')
    plt.legend(loc = 'upper left')
    plt.xlabel('p')
    plt.ylabel('qRW')
    plt.xticks(np.linspace(0.1, 0.9999, 10))
    plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
    plt.savefig(rf'Plot_qRW_phi{phi}_gamma{gamma}.pdf')
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.scatter(qRW_true, qRW_pred)
    ax.axline((0, 0), slope=1, color='black', linestyle='--')
    ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
    ax.set_xlabel('True qRW')
    ax.set_ylabel('Emulated qRW')
    plt.savefig(rf'GOF_Prediction_phi{phi}_gamma{gamma}.pdf')
    plt.show()
    plt.close()

    if LOG_RESPONSE:
        plt.plot(ps, np.log(qRW_true), 'k.-', label = 'truth')
        plt.plot(ps, np.log(qRW_pred), 'b.-', label = 'emulated')
        plt.legend(loc = 'upper left')
        plt.xlabel('p')
        plt.ylabel('log(qRW)')
        plt.xticks(np.linspace(0.1, 0.9999, 10))
        plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
        plt.savefig(rf'Plot_logqRW_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(np.log(qRW_true), np.log(qRW_pred))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
        ax.set_xlabel('True log(qRW)')
        ax.set_ylabel('Emulated log(qRW)')
        plt.savefig(rf'GOF_Prediction_log_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

    if EVGAN_RESPONSE:
        plt.plot(ps, H(qRW_true, ps), 'k.-', label = 'truth')
        plt.plot(ps, H(qRW_pred, ps), 'b.-', label = 'emulated')
        plt.legend(loc = 'upper left')
        plt.xlabel('p')
        plt.ylabel('H(qRW)')
        plt.xticks(np.linspace(0.1, 0.9999, 10))
        plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
        plt.savefig(rf'Plot_HqRW_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(H(qRW_true, ps), H(qRW_pred, ps))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
        ax.set_xlabel('True H(qRW)')
        ax.set_ylabel('Emulated H(qRW)')
        plt.savefig(rf'GOF_Prediction_H_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()



    phi      = 0.7
    gamma    = 5
    ps       = np.linspace(0.1, 0.9999, 100)
    qRW_true = qRW(ps, phi, gamma)
    qRW_pred = qRW_NN(ps, phi, gamma)

    plt.plot(ps, qRW_true, 'k.-', label = 'truth')
    plt.plot(ps, qRW_pred, 'b.-', label = 'emulated')
    plt.legend(loc = 'upper left')
    plt.xlabel('p')
    plt.ylabel('qRW')
    plt.xticks(np.linspace(0.1, 0.9999, 10))
    plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
    plt.savefig(rf'Plot_qRW_phi{phi}_gamma{gamma}.pdf')
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.scatter(qRW_true, qRW_pred)
    ax.axline((0, 0), slope=1, color='black', linestyle='--')
    ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
    ax.set_xlabel('True qRW')
    ax.set_ylabel('Emulated qRW')
    plt.savefig(rf'GOF_Prediction_phi{phi}_gamma{gamma}.pdf')
    plt.show()
    plt.close()

    if LOG_RESPONSE:
        plt.plot(ps, np.log(qRW_true), 'k.-', label = 'truth')
        plt.plot(ps, np.log(qRW_pred), 'b.-', label = 'emulated')
        plt.legend(loc = 'upper left')
        plt.xlabel('p')
        plt.ylabel('log(qRW)')
        plt.xticks(np.linspace(0.1, 0.9999, 10))
        plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
        plt.savefig(rf'Plot_logqRW_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(np.log(qRW_true), np.log(qRW_pred))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
        ax.set_xlabel('True log(qRW)')
        ax.set_ylabel('Emulated log(qRW)')
        plt.savefig(rf'GOF_Prediction_log_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

    if EVGAN_RESPONSE:
        plt.plot(ps, H(qRW_true, ps), 'k.-', label = 'truth')
        plt.plot(ps, H(qRW_pred, ps), 'b.-', label = 'emulated')
        plt.legend(loc = 'upper left')
        plt.xlabel('p')
        plt.ylabel('H(qRW)')
        plt.xticks(np.linspace(0.1, 0.9999, 10))
        plt.title(rf'qRW(p, $\phi$={phi} $\gamma$={gamma})')
        plt.savefig(rf'Plot_HqRW_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(H(qRW_true, ps), H(qRW_pred, ps))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'GOF Prediction qRW(p, $\phi$={phi} $\gamma$={gamma})')
        ax.set_xlabel('True H(qRW)')
        ax.set_ylabel('Emulated H(qRW)')
        plt.savefig(rf'GOF_Prediction_H_phi{phi}_gamma{gamma}.pdf')
        plt.show()
        plt.close()


    # Goodness of Fit plot on Validation Dataset ----------

    X_val      = np.load(rf'qRW_X_val_{N_val}.npy')
    y_val      = np.load(rf'qRW_Y_val_{N_val}.npy')
    y_val_pred = qRW_NN(X_val[:,0], X_val[:,1], X_val[:,2])

    idx        = np.where((0.1 <= X_val[:,0]) & (X_val[:,0] <= 0.9999))[0]

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.scatter(y_val[idx], y_val_pred[idx])
    ax.axline((0, 0), slope=1, color='black', linestyle='--')
    ax.set_title(rf'Goodness of Fit Plot on Validation Dataset')
    ax.set_xlabel('True qRW')
    ax.set_ylabel('Emulated qRW')
    plt.savefig(r'GOF_validation_reduced.pdf')
    plt.show()
    plt.close()

    if LOG_RESPONSE:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(np.log(y_val[idx]), np.log(y_val_pred[idx]))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'Goodness of Fit Plot on Validation Dataset')
        ax.set_xlabel('True log(qRW)')
        ax.set_ylabel('Emulated log(qRW)')
        plt.savefig(r'GOF_validation_log_reduced.pdf')
        plt.show()
        plt.close()

    if EVGAN_RESPONSE:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        ax.scatter(H(y_val[idx], X_val[:,0][idx]), H(y_val_pred[idx], X_val[:,0][idx]))
        ax.axline((0, 0), slope=1, color='black', linestyle='--')
        ax.set_title(rf'Goodness of Fit Plot on Validation Dataset')
        ax.set_xlabel('True H(qRW)')
        ax.set_ylabel('Emulated H(qRW)')
        plt.savefig(r'GOF_validation_H_reduced.pdf')
        plt.show()
        plt.close()

###############################################################################
# %% Marginal Likelihood

"""
Generate data using the dependence model and transform to GPD marginal
Use the emulated qRW() to tranform back the GPD marginals and look at
"marginal" likelihood surface along the parameters
"""

# %%
# Generate Data ---------------------------------------------------------------

data_seed = 37
np.random.seed(data_seed)
Nt         = 25
Ns         = 300

sites_xy   = np.random.random((Ns, 2)) * 10
sites_x    = sites_xy[:,0]
sites_y    = sites_xy[:,1]
minX, maxX = 0, 10
minY, maxY = 0, 10

# Knots locations w/ isometric grid ---------------------------------------
N_outer_grid             = 9
h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2,
                                        num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2,
                                        num = int(2*np.sqrt(N_outer_grid)))
x_outer_pos              = x_pos[0::2]
x_inner_pos              = x_pos[1::2]
y_outer_pos              = y_pos[0::2]
y_inner_pos              = y_pos[1::2]
X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
knots_xy                 = knots_xy[knots_id_in_domain]
knots_x                  = knots_xy[:,0]
knots_y                  = knots_xy[:,1]
k                        = len(knots_id_in_domain)

radius          = 4
effective_range = radius # Gaussian kernel effective range: exp(-3) = 0.05
bandwidth       = effective_range**2/6 # range for the gaussian kernel
range_at_knots  = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z
phi_at_knots    = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6

# Dependence Model Setup - X_star = R^phi * g(Z) ------------------------------

# Splines -------------------------------------------------

radius_from_knots = np.repeat(radius, k) # Wendland kernel influence radius from a knot

# Weight matrix generated using Gaussian Smoothing Kernel
gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# Weight matrix generated using wendland basis
wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# Covariance K for Gaussian Field g(Z) --------------------

nu        = 0.5 # exponential kernel for matern with nu = 1/2
sigsq     = 1.0 # sill for Z
sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

# Scale Mixture R^phi -------------------------------------

gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
delta = 0.0 # this is the delta in levy, stays 0
alpha = 0.5
gamma_at_knots = np.repeat(gamma, k)
gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha),
                    axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots


# Marginal Model Setup - F_Y ~ GEV(mu, sigma, xi) ---------

mu_matrix    = np.full(shape = (Ns, Nt), fill_value = 50)
sigma_matrix = np.full(shape = (Ns, Nt), fill_value = 10)
xi_matrix    = np.full(shape = (Ns, Nt), fill_value = 0.15)

# Data simulation -------------------------------------------------------------

# Gaussian Process ----------------------------------------
range_vec  = gaussian_weight_matrix @ range_at_knots
K          = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                    coords = sites_xy, kappa = nu, cov_model = "matern")
Z          = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
W          = norm_to_Pareto(Z)

# Random Scaling ------------------------------------------
phi_vec    = gaussian_weight_matrix @ phi_at_knots
R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
for t in np.arange(Nt):
    R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
R_at_sites = wendland_weight_matrix @ R_at_knots
R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in np.arange(Nt):
    R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)
X_star     = R_phi * W

# Marginal Transform --------------------------------------
if SIMULATE:
    # Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)
    # for t in np.arange(Nt):
    #     Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])

    # use multiprocessing to parallelize computation
    args_list = [(X_star[:,t], phi_vec, gamma_vec, mu_matrix[:,t], sigma_matrix[:,t], xi_matrix[:,t]) for t in range(Nt)]
    def qgev_pRW_1t(args):
        X_col, phi_vec, gamma_vec, mu_col, sigma_col, xi_col = args
        return qgev(pRW(X_col, phi_vec, gamma_vec), mu_col, sigma_col, xi_col)

    with multiprocessing.get_context('fork').Pool(processes = 6) as pool:
        results = list(tqdm(pool.imap(qgev_pRW_1t, args_list), total = Nt))
    Y = np.array(results).T

    np.save('Y_simulated.npy', Y)

# %%
# Marginal Likelihood ---------------------------------------------------------




























# %% the likelihood functions ----------------------------------------------------------------------------------------

def ll_1t_par(args):
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx = args

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)

    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])

    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

    # log likelihood of Z
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

    return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)

def ll_1t_par_NN_2p(args):
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx, \
    Ws, bs, acts = args

    # calculate X using qRW_NN_2p
    X_star = (R_vec ** phi_vec) * g(Z_vec)
    pY     = pCGP(Y, p, u_vec, scale_vec, shape_vec)
    X      = qRW_NN_2p(np.column_stack((pY, phi_vec, gamma_bar_vec, np.full((len(Y),), tau))),
                  Ws, bs, acts)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)

    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])

    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

    # log likelihood of Z
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

    return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)

def ll_1t_par_NN_2p_opt(args):
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx, \
    X, Ws, bs, acts = args

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)

    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])

    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

    # log likelihood of Z
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

    return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)

# %%
# phi -------------------------------------------------------------------------------------------------------------

# for i in range(k_phi):
for i in [0]:

    print(phi_at_knots[i]) # which phi_k value to plot a "profile" for

    lb = 0.2
    ub = 0.8
    grids = 5 # fast
    # grids = 13
    phi_grid = np.linspace(lb, ub, grids)
    phi_grid = np.sort(np.insert(phi_grid, 0, phi_at_knots[i]))

    # unchanged from above:
    #   - range_vec
    #   - K
    #   - tau
    #   - gamma_bar_vec
    #   - p
    #   - u_matrix
    #   - Scale_matrix
    #   - Shape_matrix

    # %% Optimized using qRW_NN_2p -----------------------------------------------

    """
    Idea:
        It might be much better to call NN once for a big X
        than call NN for each t separately
    """

    ll_phi_NN_2p_opt = []
    start_time = time.time()
    for phi_x in phi_grid:
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        # Calculate the X all at once
        input_list = [] # used to calculate X
        for t in range(Nt):
            pY_t = pCGP(Y[:,t], p, u_matrix[:,t], Scale_matrix[:,t], Shape_matrix[:,t])
            X_t = np.column_stack((pY_t, phi_vec_test, gamma_bar_vec, np.full((len(pY_t),), tau)))
            input_list.append(X_t)

        X_nn = qRW_NN_2p(np.vstack(input_list), Ws, bs, acts)

        # Split the X to each t, and use the
        # calculated X to calculate likelihood
        X_nn = X_nn.reshape(Nt, Ns).T

        args_list = []

        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            u_vec     = u_matrix[:,t]
            Scale_vec = Scale_matrix[:,t]
            Shape_vec = Shape_matrix[:,t]

            # copula process
            R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            logS_vec  = np.log(S_at_knots[:,t])

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            X_1t      = X_nn[:,t]

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
                            logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t,
                            X_1t, Ws, bs, acts))

        with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
            results = pool.map(ll_1t_par_NN_2p_opt, args_list)
        ll_phi_NN_2p_opt.append(np.array(results))

    ll_phi_NN_2p_opt = np.array(ll_phi_NN_2p_opt)
    np.save(rf'll_phi_NN_2p_opt_k{i}', ll_phi_NN_2p_opt)


    # %% actual calculation ------------------------------------------------------

    ll_phi     = []
    start_time = time.time()
    for phi_x in phi_grid:

        args_list = []
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            u_vec     = u_matrix[:,t]
            Scale_vec = Scale_matrix[:,t]
            Shape_vec = Shape_matrix[:,t]

            # copula process
            R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
            Z_1t      = Z[:,t]

            logS_vec  = np.log(S_at_knots[:,t])

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
                            logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t))

        with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_phi.append(np.array(results))

    ll_phi = np.array(ll_phi, dtype = object)
    np.save(rf'll_phi_k{i}', ll_phi)

    plt.plot(phi_grid, np.sum(ll_phi, axis = 1), 'b.-', label = 'actual')
    plt.plot(phi_grid, np.sum(ll_phi_NN_2p_opt, axis = 1), 'r.-', label = 'qRW_NN_2p emulator')
    plt.yscale('symlog')
    plt.axvline(x=phi_at_knots[i], color='r', linestyle='--')
    plt.legend(loc = 'upper left')
    plt.title(rf'marginal loglike against $\phi_{i}$')
    plt.xlabel(r'$\phi$')
    plt.ylabel('log likelihood')
    plt.savefig(rf'profile_ll_phi_k{i}.pdf')
    plt.show()
    plt.close()
# %%
