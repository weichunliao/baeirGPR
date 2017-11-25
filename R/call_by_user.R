
traintraintrain <- function(train_x, train_y, pred_method = "sr",
                            tune_param = FALSE, ARD = TRUE, train_all = FALSE,
                            kname = "gaussiandotrel", ktheta = NULL,
                            kbetainv = NULL, ncpu = -1, srsize = NULL,
                            tsize = 300, method = NULL) {

  train_x2 <- train_x[1:tsize,]
  train_y2 <- train_y[1:tsize,]

  if(tune_param) {
    param1 <- c(10, 2, 0.5)
    kparam <- gpr_tune(train_x2, train_y2, kernelname = "rbf",
                       init_param = param1, optim_report = 1,
                       optim_ard_report = 1, ARD = ARD, optim_ard_max = 10)
    # save(kern_param1, file = "kern_gpr_param.rdata")
  } else {
    cat("set init value for kparam\n")
    # betainv <- ???
    # thetarel <- ???
    # kname <- ???
    kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
  }

  if(pred_method == "cg_direct_lm") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "cg_direct_lm")
  } else if (pred_method == "usebigK") {
    gpr_model1 <- gpr_train(train_x = train_x, train_y = train_y, kparam = kparam,
                           in_ncpu = ncpu, method = "solve")
  } else if (pred_method == "local_gpr") {
    nclus <- nrow(ds2_trainmx) / clus_size
    nclus <- floor(nclus)
    nclus <- max(nclus, 3)
    clus1 <- data_part(ds2_trainmx, nclus = nclus, partType = "kmeans", msize = 15000)
    #clus1 <- data_part(ds2_trainmx, nclus = nclus, partType = "simple")
    csize <- unlist(lapply(clus1, length))
    cat("cluster size summary:\n")
    print(summary(csize))

    if(max(csize) > 15000) stop("max(csize) too big!\n")

    gpr_local1 <- local_gpr_train(ds2_trainmx, ds2_train_y, kern_param1, clus1, ncpu)
    gpr_model1 <- gpr_local1
  } else if (pred_method == "sr") {
    # cat("srsize, ncpu:", srsize, ncpu)
    gpr_model1 <- gpr_train_sr(train_x = train_x, train_y = train_y, kparam = kparam,
                              csize = srsize, in_ncpu = ncpu)
  }

  return (gpr_model1)
}

gpr_fit <- function() {

}
