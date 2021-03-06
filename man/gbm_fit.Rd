% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/boosting_gpr.R
\name{gbm_fit}
\alias{gbm_fit}
\title{Gradient boosting machine based on GPR}
\usage{
gbm_fit(testmx, test_y, trainmx, gbm_model, ncpu = -1)
}
\arguments{
\item{testmx}{Matrix; the features of testing dateset.}

\item{trainmx}{Matrix; the features of training data set.}

\item{gbm_model}{List; the output of gbm_train(), containing three objects:
models
pred_method
train_rmse}

\item{ncpu}{Integer; the number of thread to be used;
set to -1 to use all threads; default value is -1.}
}
\value{
List;
  prediction: the final prediction on testing dataset.
  test_rmse: numeric vector; list of rmse during testing.
}
\description{
GBM prediction
}
