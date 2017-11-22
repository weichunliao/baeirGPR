
traintraintrain <- function(train_x, train_y, pred_method = "sr",
                            tune_param = FALSE, train_all = FALSE,
                            kname = "gaussiandotrel", ktheta = NULL,
                            kbetainv = NULL,
                            ncpu = -1, srsize = NULL, method = NULL) {
  #** Number of training records to be used in Gaussian Process Regression.
  #** set to 0 if want to use everything.
  #**     for initial experiments with usebigK, set ssize=10000
  #**     for initial experiments with local_gpr, set ssize=200000
  if (train_all) {
    ssize = 0
  } else if (pred_method == "local_gpr") {
    ssize = 200000
  } else if (pred_method == "usebigK") {
    ssize = 10000
  } else if (pred_method == "sr") {
    ssize  = 50000
  } else {
    ssize = 10000
  }

  ssize = 1000
  #** parameter for local_gpr: set the cluster size
  clus_size = 10000

  #** parameter for sr: subset of regressors to be used
  # srlist = 1:10000
  srsize = 500

  #** parameter used for model tuning (tune_param = TRUE)
  #** only tsize of training points will be used for model tuning
  #tsize = 2500
  tsize = 3000
  #tsize = 5000
  #tsize = 10000
  #tsize = 1500
  #tsize = 1000

  if(tune_param) {
    param1 = c(10, 2, 0.5)
    kern_param1 = gpr_tune(ds2_train2mx, ds2_train2_y, kernelname = "rbf", init_param = param1, optim_report = 1, optim_ard_report = 1, ARD=TRUE, optim_ard_max = 10)
    save(kern_param1, file = "kern_gpr_param.rdata")
  } else {
    # load(file = "kern_gpr_param.rdata")
    cat("need init value for theta\n")
    #betainv = param$betainv
    #thetarel = param$thetarel
  # ?????????????????????? init??????????????????????????????????????
  }

  if(pred_method == "cg_direct_lm") {
    gpr_model1 = gpr_train(train_x, train_y, kern_param1, method = "cg_direct_lm")
    # cat("doing prediction\n")
    # flush.console()
    # pred1 = gpr_predict(ds2_testmx, ds2_trainmx, gpr_model1)

    # rmse_ig = sqrt(mean((ds2_test_y - pred1)^2))
    # cat(" (GPR; cg_direct_lm) rmse = ", rmse_ig, "\n")
  } else if (pred_method == "usebigK") {
    gpr_model1 = gpr_train(train_x, train_y, kern_param1)
    # cat("doing prediction\n")
    # flush.console()
    # pred1 = gpr_predict(ds2_testmx, ds2_trainmx, gpr_model1)
    # rmse1 = sqrt(mean((ds2_test_y - pred1)^2))
    # cat(" (GPR w/ARD) rmse = ", rmse1, "\n    ======\n")
    #
    # kern_param2 = list(betainv = kern_param1$betainv_noard, thetarel = kern_param1$thetarel_noard, kernelname = kern_param1$kernelname)
    # gpr_model2 = gpr_train(ds2_trainmx, ds2_train_y, kern_param2)
    # pred1 = gpr_predict(ds2_testmx, ds2_trainmx, gpr_model2)
    #
    # rmse2 = sqrt(mean((ds2_test_y - pred1)^2))
    # cat(" (GPR noard) rmse = ", rmse2, "\n")
  } else if (pred_method == "sr") {
    # cat("srsize, ncpu:", srsize, ncpu)
    gpr_model1 = gpr_train_sr(train_x, train_y, kname, ktheta, kbetainv, csize = srsize, in_ncpu = ncpu)
    # cat("doing prediction\n")
    # flush.console()
    # pred1 = gpr_predict(ds2_testmx, ds2_trainmx[srlist,], gpr_model3)
    # rmse_sr = sqrt(mean((ds2_test_y - pred1)^2))
    # cat(" (GPR_SR, ARD) rmse = ", rmse_sr, "\n    =====\n")
    #
    # kern_param2 = list(betainv = kern_param1$betainv_noard, thetarel = kern_param1$thetarel_noard, kernelname = kern_param1$kernelname)
    # gpr_model2 = gpr_train_sr(ds2_trainmx, ds2_train_y, kern_param2, srlist)
    # pred1 = gpr_predict(ds2_testmx, ds2_trainmx[srlist,], gpr_model2)
    #
    # rmse_sr2 = sqrt(mean((ds2_test_y - pred1)^2))
    # cat(" (GPR_SR noard) rmse = ", rmse_sr2, "\n")
  }

  return (gpr_model1)
}
