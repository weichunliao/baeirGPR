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
#' @param kname String; the name of kernel; default value is 'gaussiandotrel'.
#' @param ktheta Numeric vector; store kernel parameter;
#'   should be provided when tune_param is FALSE.
#' @param kbetainv Numeric; store kernel parameter betainv;
#'   shuld be provided when tune_param is FALSE.
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#' @param srsize Non-negative integer; the size of subtraining dataset;
#'   should be provided when pred_method = 'sr'
#' @param tsize Non-negative integer; parameter used for model tuning (tune_param = TRUE),
#'   only tsize of training points will be used for model tuning
#' @param clus_size Non-negative integer;
#'   parameter for local_gpr: set the cluster size;
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
                            kname = "gaussiandotrel", ktheta = NULL,
                            kbetainv = NULL, ncpu = -1, srsize = NULL,
                            clus_size = NULL) {
  if(is.null(kbetainv)) {
    cat("do not have kbetainv\n")
    return(-1)
  }
  if(is.null(ktheta)) {
    cat("do not have ktheta")
    return(-1)
  }
  cat("set init value for kparam")
  kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
  if(pred_method == "cg_direct_lm") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "cg_direct_lm")
    # gpr_model1$train_x <- train_x
  } else if (pred_method == "usebigK") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "solve")
    # gpr_model1$train_x <- train_x
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
    # gpr_model1$train_x <- train_x[gpr_model1$obslist,]
  }
  gpr_model1$pred_method <- pred_method
  return (gpr_model1)
}

#' Gaussian process regression
#'
#' GPR prediction
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
gpr_fit <- function(testmx, trainmx, gprmodel, ncpu = -1) {
  if(gprmodel$pred_method == "cg_direct_lm") {
    cat("doing prediction\n")
    flush.console()
    pred1 <- gpr_predict(testmx, trainmx, gprmodel, ncpu)
  } else if (gprmodel$pred_method == "usebigK") {
    cat("doing prediction\n")
    flush.console()
    pred1 <- gpr_predict(testmx, trainmx, gprmodel, ncpu)
  } else if (gprmodel$pred_method == "local_gpr") {
    cat("doing prediction\n")
    flush.console()
    temp <- local_gpr_predict(testmx, gprmodel, ncpu)
    pred1 <- as.matrix(temp$pred_local, ncol = 1)
  } else if (gprmodel$pred_method == "sr") {
    cat("doing prediction\n")
    flush.console()
    pred1 <- gpr_predict(testmx, trainmx, gprmodel, ncpu)
  }
  return (pred1)
}

