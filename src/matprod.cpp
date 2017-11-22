#include <Rcpp.h>
#include <vector>
#include <omp.h>
#include <time.h>
#include <string.h>
// using namespace Rcpp;
// using namespace std;

double dot(double *x1, double *x2, int size1) {
  double out1 = 0.0;
  for(int ind1 = 0; ind1 < size1; ind1++) {
    out1 = out1 + x1[ind1] * x2[ind1];
  }
  return(out1);
}


double kgaussian(double *x1, double *x2, int size1, Rcpp::NumericVector covinv, int param2) {
  double out1=0.0;

  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    x1[ind1] = fabs(x1[ind1] - x2[ind1]);
    //if(x1[ind1] < 0) x1[ind1] = x1[ind1] * -1;
  }

  for(int ind1 = 0; ind1 < size1; ind1++) {
    for(int ind2 = 0; ind2 < size1; ind2++) {
      out1 = out1 + x1[ind1] * covinv[ind1 + ind2*size1] * x1[ind2] * param2;
    }
  }

  out1 = exp(-out1);
  return(out1);
}


double kgaussians(double *x1, double *x2, int size1, Rcpp::NumericVector prec) {
  double out1 = 0.0;

  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    x1[ind1] = fabs(x1[ind1] - x2[ind1]);
    out1 = out1 + x1[ind1] * x1[ind1] * prec[ind1];
  }

  out1 = exp(-out1);
  return(out1);
}

/*
k(x1, x2) = theta0 * exp(-theta1/2 * L2norm(x1, x2)) + theta2 + theta3 x1^T * x2
out1                                       theta2    out2
*/
double kgaussiandot(double *x1, double *x2, int size1, Rcpp::NumericVector theta) {
  double out1=0.0;
  double out2=0.0;
  double tmp1;

  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    tmp1 = fabs(x1[ind1] - x2[ind1]);
    out1 = out1 + tmp1 * tmp1;
    out2 = out2 + x1[ind1] * x2[ind1] * theta[3];
  }
  out1 = exp(out1 * -0.5 * theta[1])* theta[0];
  return(out1+theta[2] + out2);
}

/*
This is the first directive of kgaussiandot for theta1.
k(x1, x2) = theta0 * exp(-theta1/2 * L2norm(x1, x2)) * -0.5 * L2norm(x1, x2)
out1
*/
double kgaussiandotgradt1(double *x1, double *x2, int size1, Rcpp::NumericVector theta) {
  double out1=0.0;
  //double out2=0.0;
  double tmp1;

  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    tmp1 = fabs(x1[ind1] - x2[ind1]);
    out1 = out1 + tmp1 * tmp1 * (-0.5);
  }
  return(theta[0] * exp(out1*theta[1])*  out1);
  //return(out1);
}

/*
This is the first directive of kgaussiandot for relevance parameter j.
k(x1, x2) = theta0 * exp(-0.5 sum_(i=1)^K rel_i * (x1i- x2i)^2) * -0.5 * (x1j- x2j)^2
out1
*/
double kgaussiandotrelgradt1(double *x1, double *x2, int size1, Rcpp::NumericVector theta, int relind) {
  double out1=0.0;
  //double out2=0.0;
  double tmp1;
  double tmp2;

  tmp2 = (x1[relind] - x2[relind]);
  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    //tmp1 = fabs(x1[ind1] - x2[ind1]);
    tmp1 = (x1[ind1] - x2[ind1]);
    out1 = out1 + tmp1 * tmp1 * (-0.5) * theta[ind1+1];
  }
  return(theta[0] * exp(out1)* -0.5* tmp2 * tmp2 );
  //return(out1);
}


/*
k(x1, x2) = theta0 * exp(-0.5 * sum_(i=1)^D theta(2+i) (x1i-x2i)^2 ) + theta1 + theta2 x1^T * x2
*/
double kgaussiandotrel(double *x1, double *x2, int size1, Rcpp::NumericVector theta) {
  double out1=0.0;
  double out2=0.0;
  double tmp1;

  //use x1 as the diff
  for(int ind1 = 0; ind1 < size1; ind1++) {
    tmp1 = fabs(x1[ind1] - x2[ind1]);
    out1 = out1 + tmp1 * tmp1 * -0.5 * theta[3+ind1];
    out2 = out2 + x1[ind1] * x2[ind1] * theta[2];
  }
  out1 = exp(out1)* theta[0];
  return(out1+theta[1] + out2);
}


/*
tcrossprod computes x %*% t(y) using FUN as the inner product operator.
allowed innername: gaussian, dot.

Gaussian kernel: exp( -1 * t(diff) * param1 * diff / param2);
diff = vec1 - vec2
param1 is a square matrix (e.g., the inverse of covariance matrix),
param2 is an scalar


*/
// t1=system.time(out1<-.Call("tcrossprod", subtrain, subtrain, 1, debug1, kparam$kernelname, kparam$thetarel, param2, ncpu))
// [[Rcpp::export]]
SEXP tcrossprod_t (Rcpp::NumericMatrix x, Rcpp::NumericMatrix y, int ifsym, int ifdebug, std::string innername, Rcpp::NumericVector param1, int param2, int ncpu_ = -1) {
  int debug = ifdebug;
  int ifsym2 = ifsym;

  // SEXP ret;
  int ret;
  // double *xp, *yp;
  int ncpu = 1;
  int maxcpu = omp_get_max_threads();
  if (ncpu_ < 1) {
    ncpu = maxcpu;
  } else if (ncpu_ < maxcpu) {
    ncpu = ncpu_;
  } else if (ncpu_ > maxcpu) {
    ncpu = maxcpu;
  }
  omp_set_num_threads(ncpu);

  std::string iname = innername;
  int ifxmat = Rf_isMatrix(x);
  int ifymat = Rf_isMatrix(y);

  if(ifxmat == 0) {
    Rprintf("x is not a matrix! stop.\n");
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  if(ifymat == 0) {
    Rprintf("y is not a matrix! stop.\n");
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  size_t xnrow = x.nrow();
  size_t xncol = x.ncol();

  size_t ynrow = y.nrow();
  size_t yncol = y.ncol();

  int kerneltype = -1;

  if(iname == "gaussian") {
    kerneltype = 2;
  } else if (iname.compare("gaussians") == 0) {
    kerneltype = 3;
  } else if (iname.compare("dot") == 0) {
    kerneltype = 1;
  } else if (iname.compare("gaussiandot") ==0) {
    kerneltype = 4;
  } else if (iname.compare("gaussiandotrel") ==0) {
    kerneltype = 5;
  } else if (iname.compare("gaussiandotgradt1") ==0) {
    kerneltype = 6;
  } else if (iname.compare("kgaussiandotrelgradt1") == 0 ) {
    kerneltype = 7;
    if (debug) Rcpp::Rcout << "param2: " << param2 << "\n";
  }  else {
    Rcpp::Rcout << "unknow kernel type " << iname << ". stop" << std::endl;
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  if (debug) {
    Rcpp::Rcout <<  "Kernel name: " << iname  << "\n";
    Rcpp::Rcout << "The dimension of x: " << xnrow << " x " << xncol << "\n";
    Rcpp::Rcout << "The dimension of y: " << ynrow << " x " << yncol << "\n";
    Rcpp::Rcout << "ifsym2 = " << ifsym2 << "\n";
  }

  double *avec1p[ncpu];
  double *avec2p[ncpu];

  for(int c1=0; c1 < ncpu; c1++) {
    double* vec1 = new double[xncol];
    memset(vec1, 0, xncol * sizeof(double));
    avec1p[c1] = vec1;

    double* vec2 = new double[xncol];
    memset(vec2, 0, xncol * sizeof(double));
    avec2p[c1] = vec2;
  }

  Rcpp::NumericMatrix outmat(xnrow, ynrow);

  if (debug) {
    Rcpp::Rcout << "size of int is " << sizeof(int) << ".\n";
  }
  for(size_t ri = 0; ri < xnrow; ri++) {
    size_t tmp9 = 0;
    if(ifsym2 == 1) {
      tmp9 = ri;
    }
  #pragma omp parallel for default(shared) num_threads(ncpu) schedule(static, 10)
    for(size_t rj=tmp9; rj<ynrow; rj++) {
      int ID = omp_get_thread_num();
      double *vec1p = avec1p[ID];
      double *vec2p = avec2p[ID];

      //copy the ri row of x to vec1 and rj row of y to vec2
      for(int tmp1=0; tmp1<xncol; tmp1++) {
        vec1p[tmp1] = x(ri, tmp1);
        vec2p[tmp1] = y(rj, tmp1);
      }

      switch(kerneltype) {
      case 1:
        outmat(ri, rj) = dot(vec1p, vec2p, xncol);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 2:
        outmat(ri, rj) = kgaussian(vec1p, vec2p, xncol, param1, param2);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 3:
        outmat(ri, rj) = kgaussians(vec1p, vec2p, xncol, param1);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 4:
        outmat(ri, rj) = kgaussiandot(vec1p, vec2p, xncol, param1);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 5:
        outmat(ri, rj) = kgaussiandotrel(vec1p, vec2p, xncol, param1);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 6:
        outmat(ri, rj) = kgaussiandotgradt1(vec1p, vec2p, xncol, param1);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      case 7:
        //need to provide the rel paramete index as well,
        outmat(ri, rj) = kgaussiandotrelgradt1(vec1p, vec2p, xncol, param1, param2-1);
        if(ifsym2 == 1) {
          outmat(rj, ri) = outmat(ri, rj);
        }
        break;
      }
    }
}

  return(outmat);
  // return (Rcpp::wrap(1));
}



//#===============
// compute kernelmat(x,y) %*% b
// [[Rcpp::export]]
SEXP kernelmdot (Rcpp::NumericMatrix x, Rcpp::NumericMatrix y, Rcpp::NumericMatrix b, double betainv, int ifdebug, std::string innername, Rcpp::NumericVector param1, int param2, int ncpu_ = -1) {
  int debug = ifdebug;

  int ifsym2 = 1;
  int ret;

  int ncpu = 1;
  int maxcpu = omp_get_max_threads();
  if (ncpu_ < 1) {
    ncpu = maxcpu;
  } else if (ncpu_ < maxcpu) {
    ncpu = ncpu_;
  } else if (ncpu_ > maxcpu) {
    ncpu = maxcpu;
  }
  omp_set_num_threads(ncpu);

  std::string iname = innername;

  int ifxmat = Rf_isMatrix(x);
  int ifymat = Rf_isMatrix(y);

  if(ifxmat == 0) {
    Rprintf("x is not a matrix! stop.\n");
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  if(ifymat == 0) {
    Rprintf("y is not a matrix! stop.\n");
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  int xnrow = x.nrow();
  int xncol = x.ncol();

  int ynrow = y.nrow();
  int yncol = x.ncol();

  int kerneltype = -1;

  if(iname=="gaussian") {
    kerneltype = 2;
  } else if (iname.compare("gaussians") ==0) {
    kerneltype = 3;
  } else if (iname.compare("dot") ==0) {
    kerneltype = 1;
  } else if (iname.compare("gaussiandot") ==0) {
    kerneltype = 4;
  } else if (iname.compare("gaussiandotrel") ==0) {
    kerneltype = 5;
  } else if (iname.compare("gaussiandotgradt1") ==0) {
    kerneltype = 6;
  } else if (iname.compare("kgaussiandotrelgradt1") == 0 ) {
    kerneltype = 7;
    if (debug) Rcpp::Rcout << "param2: " << param2 << "\n";
  }  else {
    Rcpp::Rcout << "unknow kernel type " << iname << ". stop" << std::endl;
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  if (debug) {
    Rcpp::Rcout << "Kernel name: " << iname << "\n";
    Rcpp::Rcout << "kerneltype=" << kerneltype << "\n";
    Rcpp::Rcout << "The dimension of x: " << xnrow << " x " << xncol << "\n";
    Rcpp::Rcout << "The dimension of y: " << ynrow << " x " << yncol << "\n";
  }

  double* vec1 = new double[xncol];
  memset(vec1, 0, xncol * sizeof(double));
  double* vec2 = new double[xncol];
  memset(vec2, 0, xncol * sizeof(double));

  Rcpp::NumericVector outvec(xnrow);

  double *vec1p = vec1;
  double *vec2p = vec2;

  for(int ri=0; ri<xnrow; ri++) {
  #pragma omp parallel for default(shared) num_threads(ncpu) schedule(static, 100)
    for(int rj=ri; rj<ynrow; rj++) {
      double tmpk1;
      //copy the ri row of x to vec1 and rj row of y to vec2
      for(int tmp1=0; tmp1<xncol; tmp1++) {
        vec1p[tmp1] = x(ri, tmp1);
        vec2p[tmp1] = y(rj, tmp1);
      }

      switch(kerneltype) {
      case 1:

        break;
      case 2:

        break;
      case 3:

        break;
      case 4:

        break;
      case 5:
        tmpk1 = kgaussiandotrel(vec1p, vec2p, xncol, param1);
        if(ri == rj) tmpk1 = tmpk1 + betainv;

        outvec[ri] = outvec[ri] + tmpk1 * b[rj];
        if(ri != rj) {
          outvec[rj] = outvec[rj] + tmpk1 * b(ri,0);
        }
        break;
      case 6:

        break;
      case 7:

        break;
      }
      //Rprintf("dot product of %d and %d = %f\n", ri, rj, dot(vec1p, vec2p, xncol));
    }
  }

  return(outvec);
}


//#============
//#add constant param1 to the diagonal elements of x
// [[Rcpp::export]]
SEXP diagAddConst (Rcpp::NumericMatrix x, double param1, int ifdebug) {
  int debug = ifdebug;
  int ret;

  int ifxmat = Rf_isMatrix(x);

  if(ifxmat == 0) {
    Rprintf("x is not a matrix! stop.\n");
    ret = -1;
    return(Rcpp::wrap(ret));
  }

  size_t xnrow = x.nrow();
  size_t xncol = x.ncol();

  if (debug) {
    Rprintf("The dimension of x: %d x %d\n", xnrow, xncol);
  }

  for(int ri=0; ri<xnrow; ri++) {
    x(ri, ri) = x(ri, ri) + param1;
  }

  ret = 0;
  return(Rcpp::wrap(ret));
}
