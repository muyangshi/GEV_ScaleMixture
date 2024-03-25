##################################################
# To compile on Alpine:
##################################################
# note that Alpine uses its system installed openmpi through module load
# load anaconda, activate environment, then load gcc, openmpi, gsl, and boost makes sure that
# the system mpicxx gets used and not the anaconda3/bin/mpicxx
# mpi4py is installed through using pip
module load anaconda
conda activate mcmc
module load gcc
module load openmpi
module load gsl
module load boost
# $CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas
$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas
##################################################


##################################################
# To compile on Misspiggy (misspiggy_MCMC):
##################################################
# note that misspiggy uses the conda mpicxx anyway because mpi4py, openmpi is installed through conda-forge
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gsl-2.7.1/lib:/usr/local/boost_1_79/lib # note that this line is added to my profile
export LD_LIBRARY_PATH # so no longer needed to do this
g++ -I/usr/local/gsl-2.7.1/include -I/usr/local/boost_1_79/include -std=c++11 -Wall -pedantic p_inte.cpp -shared -fPIC -L/usr/local/boost_1_79/lib -L/usr/local/gsl-2.7.1/lib -o p_inte.so -lgsl -lgslcblas
g++ -I/usr/local/gsl-2.7.1/include -I/usr/local/boost_1_79/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L/usr/local/boost_1_79/lib -L/usr/local/gsl-2.7.1/lib -o RW_inte_cpp.so -lgsl -lgslcblas
g++ -I$GSL_INC -I$BOOST_INC -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L$BOOST_LIB -L$GSL_LIB -o RW_inte_cpp.so -lgsl -lgslcblas
##################################################


##################################################
# To compile on Misspiggy using Anaconda (emulator4):
##################################################
# note that in emulator4, we installed gcc, gxx, compilers from conda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
$CXX -std=c++11 -Wall -pedantic -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lgsl -lgslcblas p_inte.cpp -shared -fPIC -o p_inte.so
##################################################


##################################################
# To compile on Mac:
##################################################
g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic p_inte.cpp -shared -fPIC -L/opt/homebrew/lib -o p_inte.so -lgsl -lgslcblas
g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas
##################################################
