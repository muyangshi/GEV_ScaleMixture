#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
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
        std::cout << "pRW_transformed failed " <<
        std::string( gsl_strerror(status)) <<
        " x: " << x << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return 1 - sqrt(gamma/(2*M_PI)) * result;
}

// no gain in accuracy, transformation is sufficient
// double pRW_transformed_2piece (double x, double phi, double gamma){
//     double result1, error1, result2, error2;
//     double params[3] = {x, phi, gamma};
//     gsl_integration_workspace * w1 = gsl_integration_workspace_alloc (10000);
//     gsl_function F1;
//     F1.function = &pRW_transformed_integrand;
//     F1.params = &params;
//     int status1 = gsl_integration_qag (&F1, 0, 0.8, 1e-12, 1e-12, 10000,
//                                     1, w1, &result1, &error1);
//     gsl_integration_workspace_free(w1);
//     if (status1) {
//         fprintf (stderr, "failed, gsl_errno1=%d\n", status1);
//     }
//     printf ("result1          = % .18f\n", result1);
//     printf ("estimated error1 = % .18f\n", error1);
//     printf ("intervals       = %zu\n", w1->size);
//     gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (10000);
//     gsl_function F2;
//     F2.function = &pRW_transformed_integrand;
//     F2.params = &params;
//     int status2 = gsl_integration_qag (&F2, 0.8, 1, 1e-12, 1e-12, 10000,
//                                     1, w2, &result2, &error2);
//     gsl_integration_workspace_free(w2);
//     if (status2) {
//         fprintf (stderr, "failed, gsl_errno2=%d\n", status2);
//     }
//     printf ("result2          = % .18f\n", result2);
//     printf ("estimated error2 = % .18f\n", error2);
//     printf ("intervals       = %zu\n", w2->size);
//     return 1 - result1 - result2;
// }

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
        std::cout << "dRW_transformed failed " <<
        std::string( gsl_strerror(status)) <<
        " x: " << x << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return result;
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
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = 0, x_hi = 2e14;
    gsl_function F;
    double params[3] = {p, phi, gamma};

    F.function = &qRW_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if(status != -2 && status != 0) 
        std::cout << "location 1 :" <<
        std::string( gsl_strerror (status) ) <<
        " p: " << p << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;

    // printf ("using %s method\n",
    //       gsl_root_fsolver_name (s));
    // printf ("%5s [%9s, %9s] %9s %9s\n",
    //       "iter", "lower", "upper", "root", "err(est)");

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            if(status != -2 && status != 0) 
                std::cout << 
                "location 2 " << 
                "p:" << p << " " << 
                "phi:" << phi << " " << 
                "gamma:" << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            if(status != -2 && status != 0) 
                std::cout <<
                "location 3 " << 
                "p:" << p << " " <<
                "phi: " << phi << " " <<
                "gamma: " << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}

// Using incomplete gamma functions for standar Pareto link function g(Z)
double upper_gamma_C(double a, double x){ // x is the integration lower bound
    return gsl_sf_gamma_inc(a, x);
}
double lower_gamma_C(double a, double x){ // x is the integration uppder bound
    return gsl_sf_gamma(a) - upper_gamma_C(a, x);
}

// likun's derivation
double dRW_standard_Pareto_C(double x, double phi, double gamma){
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(x, 1/phi)));
    return (1/pow(x,2)) * sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma;
}

// my derivation
// double dRW_standard_Pareto_C(double x, double phi, double gamma){
//     double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(x, 1/phi)));
//     double part1 = (1/pow(x,2)) * sqrt(M_1_PI) * pow(gamma/2, phi) * upper_gamma;
//     double part2 = sqrt(M_1_PI) * pow(gamma / (2 * pow(x, 1/phi)), -0.5) * 
//                     exp(-gamma / (2 * pow(x, 1/phi))) * (-gamma/(2*phi*pow(x, 1 + 1/phi)));
//     double part3 = 1 - (1/x) * pow(gamma/2, phi) * pow(gamma / (2 * pow(x, 1/phi)), -phi);
//     // std::cout << part2 << " " << part3 << std::endl;
//     return part1 - part2*part3;
// }

double pRW_standard_Pareto_C(double x, double phi, double gamma){
    double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(x, 1/phi)));
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(x, 1/phi)));
    double survival = sqrt(1/M_PI) * lower_gamma + (1/x) * sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma;
    return 1.0 - survival;
}

double qRW_standard_Pareto_to_solve(double x, void * params_ptr){
    double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return pRW_standard_Pareto_C(x, phi, gamma) - p;
}

double qRW_standard_Pareto_C_brent(double p, double phi, double gamma){
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = 1e-2, x_hi = 2e16; // pRW_standard_Pareto_C(2e16, 1, 3) = 0.99999998
    gsl_function F;
    double params[3] = {p, phi, gamma};

    F.function = &qRW_standard_Pareto_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if(status != -2 && status != 0) 
        std::cout << "location 1 :" <<
        std::string( gsl_strerror (status) ) <<
        " p: " << p << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            if(status != -2 && status != 0) 
                std::cout << 
                "location 2 " << 
                "p:" << p << " " << 
                "phi:" << phi << " " << 
                "gamma:" << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            if(status != -2 && status != 0) 
                std::cout <<
                "location 3 " << 
                "p:" << p << " " <<
                "phi: " << phi << " " <<
                "gamma: " << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}

// ---------------------------------------------------------------------------
// Convolution with a Gaussian(0, var = tau^2) nugget for threshold exceedance
// ---------------------------------------------------------------------------

double dRW_standard_Pareto_nugget_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return (1/pow(t,2)) * upper_gamma * gaussian;        
}

double dRW_standard_Pareto_nugget_C(double x, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    double lb = fmax(0.0, x - 38 * tau); // integration lowerbound for gaussian convolution
    double ub = x + 38 * tau; // integration upperbound for gaussian convolution

    // convolution of the lower gamma piece
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
    double result, error;
    double params[4] = {x, phi, gamma, tau};
    gsl_function F;
    F.function = &dRW_standard_Pareto_nugget_integrand;
    F.params = &params;
    int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
        std::cout << "dRW integration failed " <<
        std::string( gsl_strerror(status)) <<
        " x: "    << x     << " " <<
        "phi: "   << phi   << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    return sqrt(1/M_PI) * pow(gamma/2, phi) * result;
}

double pRW_standard_Pareto_nugget_lower_gamma_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(t, 1/phi)));
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return lower_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_upper_gamma_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return (1/t) * upper_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_lower_gamma_integrand_forplot(double t, double x, double phi, double gamma, double tau){
    double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(t, 1/phi)));
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return lower_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot(double t, double x, double phi, double gamma, double tau){
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return (1/t) * upper_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_C(double x, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    double lb = fmax(0.0, x - 38 * tau); // integration lowerbound for gaussian convolution
    double ub = x + 38 * tau; // integration upperbound for gaussian convolution

    // survival function of the Guassian
    double Phi_bar_x = gsl_cdf_gaussian_Q(x, tau);

    // convolution of the lower gamma piece
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
    double lower_gamma_convolution, error;
    double params[4] = {x, phi, gamma, tau};
    gsl_function F;
    F.function = &pRW_standard_Pareto_nugget_lower_gamma_integrand;
    F.params = &params;
    int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w, &lower_gamma_convolution, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
        std::cout << "lower gamma convolution failed " <<
        std::string( gsl_strerror(status)) <<
        " x: "    << x     << " " <<
        "phi: "   << phi   << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    // convolution of the upper gamma piece
    gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1e8);
    double upper_gamma_convolution, error2;
    double params2[4] = {x, phi, gamma, tau};
    gsl_function F2;
    F2.function = &pRW_standard_Pareto_nugget_upper_gamma_integrand;
    F2.params = &params2;

    int status2 = gsl_integration_qag (&F2, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w2, &upper_gamma_convolution, &error2);
    if (status2) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status2);
        std::cout << "upper gamma convolution failed " <<
        std::string( gsl_strerror(status2)) <<
        " x: "    << x     << " " <<
        "phi: "   << phi   << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w2);


    double survival = Phi_bar_x + 
                        sqrt(1/M_PI) * lower_gamma_convolution + 
                        sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma_convolution;
    return 1.0 - survival;
}

double qRW_standard_Pareto_nugget_to_solve(double x, void * params_ptr){
    double p     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    return pRW_standard_Pareto_nugget_C(x, phi, gamma, tau) - p;
}

double qRW_standard_Pareto_nugget_C_brent(double p, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = -37.0 * tau, x_hi = 1e16; // pRW_standard_Pareto_nugget_C(1e16, 1, 2, 1e3) = 1
    gsl_function F;
    double params[4] = {p, phi, gamma, tau};

    F.function = &qRW_standard_Pareto_nugget_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if(status != -2 && status != 0) 
        std::cout << "location 1 :" <<
        std::string( gsl_strerror (status) ) <<
        " p: " << p << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            if(status != -2 && status != 0) 
                std::cout << 
                "location 2 " << 
                "p:" << p << " " << 
                "phi:" << phi << " " << 
                "gamma:" << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            if(status != -2 && status != 0) 
                std::cout <<
                "location 3 " << 
                "p:" << p << " " <<
                "phi: " << phi << " " <<
                "gamma: " << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}



// double pRW_standard_Pareto_nugget_lower_gamma_transform_integrand(double s, void * params_ptr){
//     double x     = (*(double(*)[4]) params_ptr)[0];
//     double phi   = (*(double(*)[4]) params_ptr)[1];
//     double gamma = (*(double(*)[4]) params_ptr)[2];
//     double tau   = (*(double(*)[4]) params_ptr)[3];
//     double jacobian = 1/pow(s,2);
//     double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow((1-s)/s, 1/phi)));
//     double gaussian = gsl_ran_gaussian_pdf(x - (1-s)/s, tau);
//     return jacobian * lower_gamma * gaussian;
// }

// double pRW_standard_Pareto_nugget_upper_gamma_transform_integrand(double s, void * params_ptr){
//     double x     = (*(double(*)[4]) params_ptr)[0];
//     double phi   = (*(double(*)[4]) params_ptr)[1];
//     double gamma = (*(double(*)[4]) params_ptr)[2];
//     double tau   = (*(double(*)[4]) params_ptr)[3];
//     double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow((1-s)/s, 1/phi)));;
//     double gaussian = gsl_ran_gaussian_pdf(x - (1-s)/s, tau);
//     return (1/s*(1-s)) * upper_gamma * gaussian;
// }

// double pRW_standard_Pareto_nugget_transform_C(double x, double phi, double gamma, double tau){
//     gsl_set_error_handler_off();
//     double lb = fmax(0.0, 1/(x + 38 * tau + 1)); // integration lowerbound for gaussian convolution
//     double ub = fmin(1.0, 1/(x - 38 * tau + 1)); // integration upperbound for gaussian convolution
//     // survival function of the Guassian
//     double Phi_bar_x = gsl_cdf_gaussian_Q(x, tau);
//     // convolution of the lower gamma piece
//     gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
//     double lower_gamma_convolution, error;
//     double params[4] = {x, phi, gamma, tau};
//     gsl_function F;
//     F.function = &pRW_standard_Pareto_nugget_lower_gamma_transform_integrand;
//     F.params = &params;
//     int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
//                                     1, w, &lower_gamma_convolution, &error);
//     if (status) {
//         fprintf (stderr, "failed, gsl_errno=%d\n", status);
//         std::cout << "lower gamma convolution failed " <<
//         std::string( gsl_strerror(status)) <<
//         " x: "    << x     << " " <<
//         "phi: "   << phi   << " " <<
//         "gamma: " << gamma <<
//         std::endl;
//     }
//     gsl_integration_workspace_free(w);
//     // convolution of the upper gamma piece
//     gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1e8);
//     double upper_gamma_convolution, error2;
//     double params2[4] = {x, phi, gamma, tau};
//     gsl_function F2;
//     F2.function = &pRW_standard_Pareto_nugget_upper_gamma_transform_integrand;
//     F2.params = &params2;
//     int status2 = gsl_integration_qag (&F2, lb, ub, 1e-8, 1e-8, 1e8,
//                                     1, w2, &upper_gamma_convolution, &error2);
//     if (status2) {
//         fprintf (stderr, "failed, gsl_errno=%d\n", status2);
//         std::cout << "upper gamma convolution failed " <<
//         std::string( gsl_strerror(status2)) <<
//         " x: "    << x     << " " <<
//         "phi: "   << phi   << " " <<
//         "gamma: " << gamma <<
//         std::endl;
//     }
//     gsl_integration_workspace_free(w2);
//     double survival = Phi_bar_x + 
//                         sqrt(1/M_PI) * lower_gamma_convolution + 
//                         sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma_convolution;
//     return 1.0 - survival;
// }

}

int main(){
    std::cout << pRW_standard_Pareto_nugget_C(1e16, 1, 2,1000) << std::endl;
    return 0;
}
// g++ -I/opt/homebrew/include RW_inte_cpp.cpp -L/opt/homebrew/lib -o RW_inte_cpp -lgsl -lgslcblas
