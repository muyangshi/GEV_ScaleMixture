#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
// g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas

// double(*)[3] params_ptr ---- treats params_ptr as a pointer to double[3] isntead of a pointer to void
// *(double(*)[3]) params_ptr ---- derefernece the pointer
// (*(double(*)[3]) params_ptr)[2] ---- access the third element of the dereferenced double[3]
extern "C"
{
double pRW_transformed_integrand (double t, void * params_ptr) {
    double x     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    double jacobian = 1/(t*t);
    double r = (1-t)/t;
    double integrand = jacobian * pow(r, phi-1.5) * exp(-gamma/(2*r)) / (x + pow(r, phi));
    return integrand;
}

double pRW_transformed (double x, double phi, double gamma){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[3] = {x, phi, gamma};

    gsl_function F;
    F.function = &pRW_transformed_integrand;
    F.params = &params;

    int status = gsl_integration_qag (&F, 0, 1, 1e-12, 1e-12, 10000,
                                    1, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return 1 - sqrt(gamma/(2*M_PI)) * result;
}

// no gain
double pRW_transformed_2piece (double x, double phi, double gamma){
    double result1, error1, result2, error2;
    double params[3] = {x, phi, gamma};

    gsl_integration_workspace * w1 = gsl_integration_workspace_alloc (10000);
    gsl_function F1;
    F1.function = &pRW_transformed_integrand;
    F1.params = &params;
    int status1 = gsl_integration_qag (&F1, 0, 0.8, 1e-12, 1e-12, 10000,
                                    1, w1, &result1, &error1);
    gsl_integration_workspace_free(w1);
    if (status1) {
        fprintf (stderr, "failed, gsl_errno1=%d\n", status1);
    }
    printf ("result1          = % .18f\n", result1);
    printf ("estimated error1 = % .18f\n", error1);
    printf ("intervals       = %zu\n", w1->size);

    gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (10000);
    gsl_function F2;
    F2.function = &pRW_transformed_integrand;
    F2.params = &params;
    int status2 = gsl_integration_qag (&F2, 0.8, 1, 1e-12, 1e-12, 10000,
                                    1, w2, &result2, &error2);
    gsl_integration_workspace_free(w2);
    if (status2) {
        fprintf (stderr, "failed, gsl_errno2=%d\n", status2);
    }
    printf ("result2          = % .18f\n", result2);
    printf ("estimated error2 = % .18f\n", error2);
    printf ("intervals       = %zu\n", w2->size);

    return 1 - result1 - result2;
}

double dRW_transformed_integrand (double t, void * params_ptr) {
    double x     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    double jacobian = 1/(t*t);
    double integrand = jacobian * sqrt(gamma/(2*M_PI)) * 
                                    pow((1-t)/t, phi-1.5) * exp(-gamma/(2*(1-t)/t)) / pow((x + pow((1-t)/t, phi)),2);
    return integrand;
}

double dRW_transformed (double x, double phi, double gamma){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[3] = {x, phi, gamma};

    gsl_function F;
    F.function = &dRW_transformed_integrand;
    F.params = &params;

    int status = gsl_integration_qag (&F, 0, 1, 1e-12, 1e-12, 10000,
                                    1, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return 1 - result;
}

double qRW_to_solve (double x, void * params_ptr) {
    double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return pRW_transformed(x, phi, gamma) - p;
}
double qRW_to_solve_df (double x, void * params_ptr){
    // double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return dRW_transformed(x, phi, gamma);
}
void qRW_to_solve_fdf (double x, void * params_ptr, double * f, double * df){
    *f = qRW_to_solve(x, params_ptr);
    *df = qRW_to_solve_df(x, params_ptr);
}

double qRW_transformed_brent (double p, double phi, double gamma){
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = 0.01, x_hi = 2e14;
    gsl_function F;
    double params[4] = {p, phi, gamma};

    F.function = &qRW_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    // printf ("using %s method\n",
    //       gsl_root_fsolver_name (s));
    // printf ("%5s [%9s, %9s] %9s %9s\n",
    //       "iter", "lower", "upper", "root", "err(est)");

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);

            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free (s);

    return r;
}

}