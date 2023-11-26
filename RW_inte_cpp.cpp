#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
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
    double integrand = jacobian * sqrt(gamma/(2*M_PI)) * 
                                    pow((1-t)/t, phi-1.5) * exp(-gamma/(2*(1-t)/t)) / (x + pow((1-t)/t, phi));
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
                                    2, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
    }
    gsl_integration_workspace_free(w);

    printf ("result          = % .18f\n", result);
    printf ("estimated error = % .18f\n", error);
    printf ("intervals       = %zu\n", w->size);

    return 1 - result;
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
                                    2, w1, &result1, &error1);
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
                                    2, w2, &result2, &error2);
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
                                    2, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
    }
    gsl_integration_workspace_free(w);

    printf ("result          = % .18f\n", result);
    printf ("estimated error = % .18f\n", error);
    printf ("intervals       = %zu\n", w->size);

    return 1 - result;
}

}