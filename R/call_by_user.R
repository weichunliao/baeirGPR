#' Gaussian process regression
#'
#' Training a gaussian process regression model
#'
#' @param train_x Matrix; the features of training data set.
#' @param train_y Matrix; y.
#' @param pred_method String; Set the model training approach.
#'   cg_direct_lm: conjugate gradient decent with no preconditioning
#'   usebigK: use built-in matrix inversion
#'   local_gpr: use Local GPR method
#'   sr: subset regressors
#' @param tune_param Boolean; Set to TRUE to tune parameters of kernel function
#'   default value is TRUE.
#' @param ARD Boolean; set to TRUE to use ARD when tuning param;
#'   default value is TRUE.
#' @param kname String; the name of kernel; default value is 'gaussiandotrel'.
#' @param ktheta Numeric vector; store kernel parameter;
#'   should be provided when tune_param is FALSE.
#' @param kbetainv Numeric; store kernel parameter betainv;
#'   shuld be provided when tune_param is FALSE.
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#' @param srsize Integer; the size of subtraining dataset;
#'   should be provided when pred_method = 'sr'
#' @param tsize Integer; parameter used for model tuning (tune_param = TRUE),
#'   only tsize of training points will be used for model tuning
#' @param clus_size Integer; parameter for local_gpr: set the cluster size;
#'   should be provided when pred_method = 'local_gpr'.
#'
#' @return  return a list having four objects:
#'   alpha
#'   kparam
#'   train_x
#'   pred_method
#'
#' @export
traintraintrain <- function(train_x, train_y, pred_method = "sr",
                            tune_param = TRUE, ARD = TRUE,
                            kname = "gaussiandotrel", ktheta = NULL,
                            kbetainv = NULL, ncpu = -1, srsize = NULL,
                            tsize = NULL, clus_size = NULL) {

  if(tune_param) {
    if (is.null(tsize)) {
      cat("the tsize is not provided.")
      return (-1)
    }

    train_x2 <- train_x[1:tsize,]
    train_y2 <- train_y[1:tsize,]

    param1 <- c(10, 2, 0.5)
    kparam <- gpr_tune(train_x2, train_y2, kernelname = "rbf",
                       init_param = param1, optim_report = 1,
                       optim_ard_report = 1, ARD = ARD, optim_ard_max = 10,
                       in_ncpu = ncpu)
  } else {
    cat("set init value for kparam\n")
    kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
  }

  if(pred_method == "cg_direct_lm") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "cg_direct_lm")
    gpr_model1$train_x <- train_x

  } else if (pred_method == "usebigK") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "solve")
    gpr_model1$train_x <- train_x

  } else if (pred_method == "local_gpr") {
    nclus <- nrow(ds2_trainmx) / clus_size
    nclus <- floor(nclus)
    nclus <- max(nclus, 3)
    clus1 <- data_part(ds2_trainmx, nclus = nclus, partType = "kmeans", msize = 15000)
    csize <- unlist(lapply(clus1, length))
    cat("cluster size summary:\n")
    print(summary(csize))

    if(max(csize) > 15000) stop("max(csize) too big!\n")

    gpr_model1 <- local_gpr_train(ds2_trainmx, ds2_train_y, kparam, clus1, ncpu)
    gpr_model1$train_x <- train_x

  } else if (pred_method == "sr") {
    # cat("srsize, ncpu:", srsize, ncpu)
    gpr_model1 <- gpr_train_sr(train_x = train_x, train_y = train_y, kparam = kparam,
                              csize = srsize, in_ncpu = ncpu)
    gpr_model1$train_x <- train_x[c(1:srsize),]

  }

  gpr_model1$pred_method <- pred_method
  return (gpr_model1)
}

#' Gaussian process regression
#'
#' @param testmx Matrix; the features of testing dateset
#' @param gprmodel List; the output of traintraintrain(), containing four objects:
#'   alpha
#'   kparam
#'   train_x
#'   pred_method
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#'
#' @return the prediction of testing dataset.
#'
#' @export
gpr_fit <- function(testmx, gprmodel, ncpu = -1) {
  if(gprmodel$pred_method == "cg_direct_lm") {
    cat("doing prediction\n")
    flush.console()
    pred1 = gpr_predict(testmx, gprmodel, ncpu)

  } else if (gprmodel$pred_method == "usebigK") {
    cat("doing prediction\n")
    flush.console()
    pred1 = gpr_predict(testmx, gprmodel, ncpu)

  } else if (gprmodel$pred_method == "local_gpr") {
    cat("doing prediction\n")
    flush.console()
    pred1 = local_gpr_predict(testmx, gprmodel, ncpu)

  } else if (gprmodel$pred_method == "sr") {
    cat("doing prediction\n")
    flush.console()
    pred1 = gpr_predict(testmx, gprmodel, ncpu)

  }
  return (pred1)
}
