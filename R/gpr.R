#to be called by users
#init_param = c(betainv, theta0, theta1)
gpr_tune <- function (x, y, kernelname = "rbf", ARD = TRUE,
                      init_param = c(10, 2, 0.5), optim_rbf_max = 100,
                      optim_trace = 1, optim_report = 5, optim_ard_max = 15,
                      optim_ard_trace = 1, optim_ard_report = 5) {

  if(kernelname == "rbf") {
    kname <- "gaussiandotrel"
  } else {
    stop("gpr_turn: Unknown kernelname", kernelname, "\n")
  }
  cat("gpr_tune: optimizing marginal log-likelihood\n")

  cat("    Using initial value [betainv theta0 theta1]:", param1, "\n")
  flush.console()
  #param1 = c(10, 2, 0.5)
  param1 <- init_param
  nfeature <- ncol(x)
  ans1 <- optim(param1, loglike_gauss, grad_gauss_loglike, method = "L-BFGS-B", lower = c(1e-6, 1e-6, 1e-6), upper = c(Inf, Inf, Inf), control = list(maxit = optim_rbf_max, trace = optim_trace, REPORT = optim_report, fnscale = -1.0), ds2_train2mx = x, train_y = y)
  param1 <- c((ans1$par)[2:3], 0, 0)
  betainv <- (ans1$par)[1]

  cat("[NOARD] Optimal kernel parameter [theta0 theta1]=", param1[1:2], "\n")
  cat("[NOARD] Optimal kernel parameter betainv=", betainv, "\n")

  cat("First stage optim (rbf, no ARD) convergence (0: success; 1: maxit reached)=", ans1$convergence, "msg=", ans1$message, "\n")
  flush.console()

  if(ARD == FALSE) {
    thetarel <- c(param1[1], 0, 0, rep(param1[2], nfeature))
    outparam <- list(betainv = betainv, thetarel =thetarel, nfeature = nfeature, kernelname = kname, ARD = FALSE)
  } else {
    cat("Running second stage optim (ARD)\n")

    thetarel_noard <- c(param1[1], 0, 0, rep(param1[2], nfeature))
    betainv_noard <- betainv

    flush.console()
    #kname <- "gaussiandotrel"
    #
    rel <- c(betainv, param1[1],  rep(param1[2], nfeature))
    #ll2=loglikerel2(rel)
    #
    ans2 <- optim(rel, loglikerel2, grad_gauss_loglikerel2, method = "L-BFGS-B", lower = rep(1e-6, length(rel)), upper = rep(Inf, length(rel)), control = list(maxit = optim_ard_max, trace = optim_ard_trace, REPORT = optim_ard_report, fnscale = -1.0), ds2_train2mx = x, train_y = y)
    #
    cat("Second stage optim (RBF with ARD) convergence (0: success; 1: maxit reached)=", ans2$convergence, "msg=", ans2$message, "\n")
    flush.console()

    betainv <- ans2$par[1]
    thetarel <- c(ans2$par[2], 0, 0, ans2$par[3:length(ans2$par)])
    outparam <- list(betainv = betainv, thetarel = thetarel, nfeature = nfeature, kernelname = kname, ARD = TRUE, thetarel_noard = thetarel_noard, betainv_noard = betainv_noard)

    cat("Optimal kernel parameter (aRD) betainv=", betainv, "\n")
    cat("Optimal kernel parameter (ARD) [theta0]=", ans2$par[2], "\n")
    cat("ARD param (head)=", head(thetarel[4:length(thetarel)], n = 10), "\n")
  }
  return(outparam)
}

#tuning using local models
gpr_tune_lm <- function(x, y, kernelname = "rbf", ARD = TRUE,
                        init_param = c(10, 2, 0.5), clusters = NULL,
                        optim_rbf_max = 100, optim_trace = 1,
                        optim_report = 5, optim_ard_max = 20,
                        optim_ard_trace = 1, optim_ard_report = 5,
                        ncput = -1) {

  if(kernelname == "rbf") {
    kname <- "gaussiandotrel"
  } else {
    stop("gpr_turn: Unknown kernelname", kernelname, "\n")
  }
  cat("gpr_tune_lm: optimizing marginal log-likelihood\n")

  cat("    Using initial value [betainv theta0 theta1]:", param1, "\n")
  flush.console()
  #param1 <- c(10, 2, 0.5)
  param1 <- init_param
  nfeature <- ncol(x)

  clus_x <- list()
  clus_y <- list()
  cc <- 0
  for(alist in clusters) {
    cc <- cc + 1
    clus_x[[cc]] <- x[alist,]
    clus_y[[cc]] <- y[alist,]
  }


  ans1 <- optim(param1, loglike_gauss_clus, grad_gauss_loglike_clus, method = "L-BFGS-B", lower = c(1e-6, 1e-6, 1e-6), upper = c(Inf, Inf, Inf), control = list(maxit = optim_rbf_max, trace = optim_trace, REPORT = optim_report, fnscale = -1.0), train_x_list = clus_x, train_y_list = clus_y)
  param1 <- c((ans1$par)[2:3], 0, 0)
  betainv <- (ans1$par)[1]

  cat("Optimal kernel parameter [theta0 theta1]=", param1[1:2], "\n")
  cat("Optimal kernel parameter betainv=", betainv, "\n")

  cat("First stage optim (rbf, no ARD) convergence (0: success; 1: maxit reached)=", ans1$convergence, "msg=", ans1$message, "\n")
  flush.console()

  if(ARD == FALSE) {
    thetarel <- c(param1[1], 0, 0, rep(param1[2], nfeature))
    outparam <- list(betainv = betainv, thetarel = thetarel, nfeature = nfeature, kernelname = kname, ARD = FALSE)
  } else {
    cat("Running second stage optim (ARD)\n")
    stop("not finished yet.")

    thetarel_noard <- c(param1[1], 0, 0, rep(param1[2], nfeature))
    betainv_noard <- betainv

    flush.console()
    #kname = "gaussiandotrel"
    #
    rel <- c(betainv, param1[1], rep(param1[2], nfeature))
    #ll2=loglikerel2(rel)
    #
    ans2 <- optim(rel, loglikerel2, grad_gauss_loglikerel2, method = "L-BFGS-B", lower = rep(1e-6, length(rel)), upper = rep(Inf, length(rel)), control = list(maxit = optim_ard_max, trace = optim_ard_trace, REPORT = optim_ard_report, fnscale = -1.0), ds2_train2mx = x, train_y = y)
    #
    cat("Second stage optim (RBF with ARD) convergence (0: success; 1: maxit reached)=", ans2$convergence, "msg=", ans2$message, "\n")
    flush.console()

    betainv <- ans2$par[1]
    thetarel <- c(ans2$par[2], 0, 0, ans2$par[3:length(ans2$par)])
    outparam <- list(betainv = betainv, thetarel = thetarel, nfeature = nfeature, kernelname = kname, ARD = TRUE, thetarel_noard = thetarel_noard, betainv_noard = betainv_noard)

  }
  return(outparam)
}


#cluster data and return a list of row indces of each cluster.
data_part <- function(x, partType="kmeans", nclus=10, iter.max = 100, msize = 13000) {
  #partType = "simple"
  #partType = "kmeans"

  if(partType == "kmeans") {
    cat("reorder observations via kmeans\n")
    flush.console()

    cat("initialize kmeans..\n")
    #msize = 13000
    #alg = "Forgy"
    #alg = "MacQueen"
    alg <- "Hartigan-Wong"
    #nstep = 10
    km_ret1 <- kmeans(x, nclus, algorithm = alg, iter.max)

    clusIndex <- km_ret1$cluster
    clusters <- 1:nclus
    #obsClusters_old = lapply(clusters,function(y){which(y==clusIndex)})
    obsClusters <- lapply(clusters, function(y){ which(y==clusIndex) })

    obsClusters2 <- list()
    lid <- 0
    for (ii in 1:nclus) {
      if (length(obsClusters[[ii]]) < msize) {
        lid <- lid + 1
        obsClusters2[[lid]] <- obsClusters[[ii]]
      } else {
        nc <- floor(length(obsClusters[[ii]]) / (msize / 2.1))
        cind1 <- cut(sample(seq(1, length(obsClusters[[ii]]))), breaks = nc, labels = F)
        for (j3 in 1:nc) {
          lid <- lid + 1
          obsClusters2[[lid]] <- obsClusters[[ii]][cind1 == j3]
        }
      }
    }

    obsClusters <- obsClusters2

  } else if (partType == "simple") {
    clusIndex <- cut(seq(1,nrow(x)), breaks = nclus, labels = F)
    obsClusters <- lapply(1:nclus, function(y){ which(y == clusIndex) })
  } else {
    max1 <- abs(max(bigK)) * 1.1
    dist1 <- matrix(max1, nrow = nrow(bigK), ncol = ncol(bigK)) - bigK

    stop("not supported")
  }

  return(obsClusters)
}

#method = c("solve", "cg_direct", "cg_ichol")
gpr_train <- function(train_x, train_y, kparam, method = "solve", ig_tol = 0.01, ncpu = -1) {
  #now, construct the full matrix bigK
  debug1 <- 0
  param2 <- 1
  nobs <- nrow(train_x)
  cat("GPR training with", nobs, "data points\n")
  flush.console()

  if(method != "cg_direct_lm") {
    cat("constructing bigK\n")
    flush.console()
    t1 <- system.time(bigK <- tcrossprod_t(train_x, train_x, 1, debug1, kparam$kernelname, kparam$thetarel, param2, ncpu))
    cat("bigK consumed time:\n")
    print(t1)
    flush.console()
    diagAddConst(bigK, kparam$betainv, 1)
  }

  #=====
  #now, construct bigMinv by doing incomplete choloskey decomposition, and find the inverse of its cross product.

  #ifrunicol = TRUE
  #ifrunicol = FALSE

  if(method == "cg_ichol") {
    L1 <- bigK * 1.0

    cat("performing incomplete choleskey decomposition\n")
    flush.console()
    t1 <- system.time(a1 <- ichol(L1, 1))
    cat('ichol cnosumed time:\n')
    print(t1)
    flush.console()

    cat("doing chol2inv\n")
    t1 <- system.time(bigMinv <- chol2inv(t(L1)))
    cat("   chol2inv consumed time:\n")
    print(t1)
  }

  if(method == "cg_ichol") {
    cat("running CG with preconditioner\n")
    xvec <- matrix(0, nrow=nrow(bigK), ncol=1)
    #xvec = alpha2
    rvec <- as.matrix(train_y, ncol=1) - bigK %*% xvec
    zvec <- bigMinv %*% rvec
    pvec <- zvec

    for(kind in 1:nrow(bigK)) {
      cat("At iteration", kind, "\n")
      flush.console()

      ak <- sum(rvec * zvec) / (t(pvec) %*% bigK %*% pvec)
      ak <- drop(ak)
      xvec2 <- xvec + ak * pvec
      rvec2 <- rvec - ak * bigK %*% pvec
      rvec2_norm <- sqrt(mean(rvec2 * rvec2))
      cat("rvec2_norm = ", rvec2_norm, "\n")

      zvec2 <- bigMinv %*% rvec2
      bk <- (t(zvec2) %*% rvec2) / (t(zvec) %*% rvec)
      bk <- drop(bk)
      pvec2 <- zvec2 + bk * pvec

      #=== iterate
      zvec <- zvec2
      rvec <- rvec2
      pvec <- pvec2
      xvec <- xvec2

      if(rvec2_norm < ig_tol) {
        cat("converged!\n")
        break
      }
    }
  } else if (method == "cg_direct") {
    cat("running CG without preconditioner\n")

    tcg1 <- proc.time()

    xvec <- matrix(0, nrow = nrow(bigK), ncol = 1)
    #xvec = alpha2
    rvec <- as.matrix(train_y, ncol = 1) - bigK %*% xvec
    zvec <- rvec
    pvec <- zvec

    for(kind in 1:nrow(bigK)) {
      cat("At iteration", kind, "\n")
      flush.console()

      ak <- sum(rvec * zvec) / (t(pvec) %*% bigK %*% pvec)
      ak <- drop(ak)
      xvec2 <- xvec + ak * pvec
      rvec2 <- rvec - ak * bigK %*% pvec
      rvec2_norm <- sqrt(mean(rvec2 * rvec2))
      cat("rvec2_norm = ", rvec2_norm, "\n")

      zvec2 <- rvec2
      bk <- (t(zvec2) %*% rvec2) / (t(zvec) %*% rvec)
      bk <- drop(bk)
      pvec2 <- zvec2 + bk * pvec

      #=== iterate
      zvec <- zvec2
      rvec <- rvec2
      pvec <- pvec2
      xvec <- xvec2
      #if(kind %% 10 == 0) print(lsos())

      if(rvec2_norm < ig_tol) {
        cat("converged!\n")
        break
      }
    }

    tcg2 <- proc.time()
    cat("CG takes time:\n")
    print(tcg2 - tcg1)


  } else if (method == "solve") {
    xvec <- solve(bigK, train_y)
  } else if (method == "cg_direct_lm") {
    cat("running CG without preconditioner\n")
    flush.console()
    tcg1 <- proc.time()

    xvec <- matrix(0, nrow=nobs, ncol=1)

    cat("    preparing IG...\n")
    flush.console()
    t1 <- system.time(rvec <- as.matrix(train_y, ncol = 1) - as.matrix(kernelmdot(train_x, train_x, xvec, kparam$betainv, debug1, kparam$kernelname, kparam$thetarel, param2), ncol = 1))
    cat("    kernelM %*% xvec consumes time:\n")
    print(t1)
    flush.console()

    zvec <- rvec
    pvec <- zvec

    cat("    start IG iteration...\n")
    flush.console()
    for(kind in 1:nobs) {
      cat("At iteration", kind, "\n")
      flush.console()

      # bigK %*% pvec
      bigKpvec <- as.matrix(kernelmdot(train_x, train_x, pvec, kparam$betainv, debug1, kparam$kernelname, kparam$thetarel, param2), ncol=1)
      ak <- sum(rvec * zvec) / (t(pvec) %*% bigKpvec)
      ak <- drop(ak)
      xvec2 <- xvec + ak * pvec
      # bigK %*% pvec
      rvec2 <- rvec - ak * bigKpvec
      rvec2_norm <- sqrt(mean(rvec2 * rvec2))
      cat("rvec2_norm = ", rvec2_norm, "\n")

      zvec2 <- rvec2
      bk <- (t(zvec2) %*% rvec2) / (t(zvec) %*% rvec)
      bk <- drop(bk)
      pvec2 <- zvec2 + bk * pvec

      #=== iterate
      zvec <- zvec2
      rvec <- rvec2
      pvec <- pvec2
      xvec <- xvec2
      #if(kind %% 10 == 0) print(lsos())

      if(rvec2_norm < ig_tol) {
        cat("converged!\n")
        break
      }
    }

    tcg2 <- proc.time()
    cat("CG takes time:\n")
    print(tcg2 - tcg1)

  } else {
    stop("Unknow method", method,)
  }

  return(list(alpha = xvec, kparam = kparam))
}

gpr_predict <- function(testmx, trainmx, gprmodel, ncpu = -1) {
  debug1 <- 0
  param2 <- 1

  t2 <- system.time(Ksmall2 <- tcrossprod_t(testmx, trainmx, 0, debug1, gprmodel$kparam$kernelname, gprmodel$kparam$thetarel, param2, ncpu))
  cat("Ksmall2 consumed time:\n")
  print(t2)

  pred1 <- Ksmall2 %*% gprmodel$alpha
  return(pred1)
}


gpr_train_sr <- function(train_x, train_y, kname, ktheta, kbetainv, csize, in_ncpu) {
  cat("gpr_sr trainer\n")

  obslist <- c(1:csize)
  cat("gpr_train_sr: subset size=", csize, "\n")
  cat("training data size=", nrow(train_x), "\n")
  flush.console()


  subtrain <- train_x[obslist, ]
  debug1 <- 0
  param2 <- 1
  t1 <- system.time(out1 <- tcrossprod_t(subtrain, subtrain, 1, debug1, kname, ktheta, param2, in_ncpu))
  cat("out1 consumed time:\n")
  print(t1)

  debug1 <- 0
  sym2 <- 0
  cat("compute kmn\n")
  flush.console()
  t1 <- system.time(kmn <- tcrossprod_t(subtrain, train_x, sym2, debug1, kname, ktheta, param2, in_ncpu))

  cat("kmn consumed time:\n")
  print(t1)
  flush.console()

  t1 <- system.time(kmn_cross <- tcrossprod(kmn))
  cat("tcrossprod consumed time:\n")
  print(t1)

  alpha <- solve((kmn_cross + kbetainv * out1), kmn %*% train_y)

  kparam <- list(betainv = kbetainv, kernelname = kname, theta = ktheta)
  return(list(alpha = alpha, kparam = kparam))
}


# =======================
local_gpr_train <- function(train_x, train_y, kparam, obsClusters, ncpu = 4) {
  nclus <- length(obsClusters)
  # cc=0
  alphaList <- list()
  length(alphaList) <- nclus
  CninvList <- list()
  length(CninvList) <- nclus

  for(ag in 1:nclus) {
    cat("processing group", ag, "\n")
    # cc = cc+1
    obslist <- obsClusters[[ag]]
    csize <- length(obslist)
    cat("  group size=", csize, "\n")
    if(csize > 1) {
      flush.console()

      subtrain <- train_x[obslist,]
      ifsym <- 1
      debug1 <- 0
      param2 <- 1
      t1 <- system.time(out1 <- tcrossprod_t(subtrain, subtrain, ifsym, debug1, kparam$kernelname, kparam$thetarel, param2, ncpu))###ncpu
      cat("out1 consumed time:\n")
      print(t1)
      Cn <- out1 + kparam$betainv * diag(nrow(out1))
      Cninv <- solve(Cn)
      CninvList[[ag]] <- Cninv
      alpha  <- Cninv %*% train_y[obslist]
      alphaList[[ag]] <- alpha
    } else {
      alphaList[[ag]] <- NULL
    }
  }
  return(list(alphaList = alphaList, kparam = kparam, obsClusters = obsClusters, nclus = nclus, CninvList = CninvList))
}


local_gpr_predict <- function (test_x, train_x, lgpr_model) {

  if(nrow(test_x) > 3000) {
    nclus <- nrow(test_x) / 2000
    clus1 <- data_part(test_x, nclus = nclus, partType = "simple")
  } else {
    clus1 <- list()
    clus1[[1]] <- 1:nrow(test_x)
  }

  pred_llist <- list()
  lc <- 0
  for(testobs in clus1) {
    lc <- lc + 1
    ntest <- length(testobs)
    thistest <- test_x[testobs,]
    #todo... gen test data...

    obsClusters <- lgpr_model$obsClusters
    nclus <- lgpr_model$nclus
    alphaList <- lgpr_model$alphaList
    kttList <- list()
    length(kttList) <- nclus
    debug1 <- 0
    param2 <- 1
    for(ag in 1:nclus) {
      cat(lc, "of", length(clus1), "local_gpr_predict: ksmall: processing group", ag, "\n")
      flush.console()
      obslist <- obsClusters[[ag]]

      if(length(obslist) > 1) {
        t2 <- system.time(Ksmall2 <- tcrossprod_t(thistest, train_x[obslist,], 0, debug1, lgpr_model$kparam$kernelname, lgpr_model$kparam$thetarel, param2))
        #cat("Ksmall2 consumed time:\n")
        #print(t2)
        kttList[[ag]] <- Ksmall2
      } else {
        Ksmall2 <- matrix(-Inf, nrow = ntest, ncol = length(obslist))
        kttList[[ag]] <- Ksmall2
      }
    }

    #make prediction
    predAll <- list()
    for (ag in 1:nclus) {
      cat(lc, "of", length(clus1), "local_gpr_predict: predict local: processing group", ag, "\n")
      flush.console()
      if (is.null(alphaList[[ag]])) {
        predAll[[ag]] <- rep(0, ntest)
      } else {
        pred1 <- kttList[[ag]] %*% alphaList[[ag]]
        predAll[[ag]] <- pred1
      }
    }

    kttcomp <- matrix(NA, nrow = ntest, ncol = nclus)
    for (ag in 1:nclus) {
      cat(lc, "of", length(clus1), "local_gpr_predict: localmax: processing group", ag, "\n")
      flush.console()
      kttcomp[,ag] <- apply(kttList[[ag]], 1, max)
    }

    clusSel <- apply(kttcomp, 1, which.max)

    predBC <- matrix(NA, nrow=ntest, ncol=1)
    for (ii in 1:ntest) {
      predBC[ii] <- predAll[[clusSel[ii]]][ii]
    }

    dobcm <- FALSE
    if (dobcm) {
      #single point bcm
      predSdList <- list()
      length(predSdList) <- nclus
      t2 <- system.time(kc <- tcrossprod_t(test_x, test_x, 1, debug1, lgpr_model$kparam$kernelname, lgpr_model$kparam$thetarel, param2))
      qvar <- 0
      bcmPred <- 0
      for (ag in 1:nclus) {
        cat("local_gpr_predict: BCM variance: processing group", ag, "\n")
        flush.console()

        cov1 <- kttList[[ag]] %*% lgpr_model$CninvList[[ag]] %*% t(kttList[[ag]])
        var1 <- diag(kc) - diag(cov1)
        predSdList[[ag]] <- sqrt(var1)
        qvar <- qvar + 1 / var1
        bcmPred <- bcmPred + predAll[[ag]] / var1
      }

      qvar <- 1 / (qvar - (nclus-1) / diag(kc))
      bcmPred <- bcmPred * qvar
    } else {
      bcmPred <- NULL
    }

    pred_llist[[lc]] <- predBC
  }
  return(list(pred_local = unlist(pred_llist), pred_sbcm = bcmPred))
}



# =========================

loglike_gauss_clus <- function(theta, train_x_list, train_y_list) {
  listn <- length(train_x_list)

  sum1 <- 0.0
  for (cc in 1:listn) {
    sum1 <- sum1 + loglike_gauss(theta, train_x_list[[cc]], train_y_list[[cc]])
  }
  return(sum1)
}

grad_gauss_loglike_clus <- function(theta, train_x_list, train_y_list) {
  listn <- length(train_x_list)

  sum1 <- 0.0
  for(cc in 1:listn) {
    sum1 <- sum1 + grad_gauss_loglike(theta, train_x_list[[cc]], train_y_list[[cc]])
  }
  return(sum1)
}


# gaussian kernels
# theta [betainv theta1]
loglike_gauss <- function(theta, ds2_train2mx, train_y) {
  kname = "gaussiandot"

  ifsym <- 1
  debug1 <- 0
  theta1 <- c(theta[2:3], 0, 0)
  betainv <- theta[1]
  param2 <- 1

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  logdet1 <- -0.5 * determinant(out1, logarithm = TRUE)$modulus[1]
  ip1 <- - 0.5 * t(train_y) %*% out1inv %*% train_y
  ret1 <- logdet1 +ip1
  flush.console()
  # cat("   loglike: theta=", theta1, "betainv=",betainv, "ret1=", ret1, "logdet1=", logdet1, "ip1=", ip1, "\n")
  return(ret1)
}


grad_gauss_loglike <- function(theta, ds2_train2mx, train_y) {
  debug1 <- 0
  ifsym <- 1
  kname <- "gaussiandot"

  theta1 <- c(theta[2:3], 0, 0)
  betainv <- theta[1]
  param2 <- 1

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  # compute grad_theta0
  tmp1 <- c(1, theta1[2], 0, 0)
  cn_ptheta0 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, tmp1, param2)

  tmpkname <- "gaussiandotgradt1"
  tmp1 <- c(theta1[1], theta1[2], 0,0)
  cn_ptheta1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, tmpkname, tmp1, param2)

  grad0 <- -0.5 * sum(diag(out1inv %*% cn_ptheta0))  + 0.5 * t(train_y) %*% out1inv %*% cn_ptheta0 %*% out1inv %*% train_y
  grad1 <- -0.5 * sum(diag(out1inv %*% cn_ptheta1))  + 0.5 * t(train_y) %*% out1inv %*% cn_ptheta1 %*% out1inv %*% train_y
  gradbetainv <- -0.5 * sum(diag(out1inv))  + 0.5 * t(train_y) %*% out1inv %*% out1inv %*% train_y

  return(c(gradbetainv, grad0, grad1))
}


# marginal log-likelihood using gaussian kernel with individual relevance parameters for input features
# theta = [betainv rel]
loglikerel2 <- function(theta, ds2_train2mx, train_y) {
  debug1 <- 0
  ifsym <- 1
  param2 <- 1

  tmp1 <- theta
  kname <- "gaussiandotrel"

  theta1 <- c(tmp1[2], 0, 0, tmp1[3:length(tmp1)])
  betainv <- tmp1[1]

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  out1 <- out1 + betainv * diag(nrow(out1))
  # lambda = 1/100000
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  logdet1 <- -0.5 * determinant(out1, logarithm = TRUE)$modulus[1]
  ip1 <- - 0.5 * t(ds2_train2$y) %*% out1inv %*% ds2_train2$y
  ret1 <- logdet1 + ip1
  flush.console()
  # cat("   loglike: theta=", theta1[1:3], "betainv=",betainv, "ret1=", ret1, "logdet1=", logdet1, "ip1=", ip1, "\n")
  # cat("         theta[4:]=", theta1[4:length(theta1)], "\n")
  return(ret1)
}


grad_gauss_loglikerel2 <- function(theta, ds2_train2mx, train_y) {
  debug1 <- 0
  kname <- "gaussiandotrel"
  ifsym <- 1
  param2 <- 1

  theta1 <- c(theta[2], 0, 0, theta[3:length(theta)])
  betainv <- theta[1]

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  gradbetainv <- -0.5 * sum(diag(out1inv)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% out1inv %*% ds2_train2$y

  # compute grad_theta0
  tmp1 <- c(1, 0, 0, theta1[4:length(theta1)])
  cn_ptheta0 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, tmp1, param2)
  grad0 <- -0.5 * sum(diag(out1inv %*% cn_ptheta0)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% cn_ptheta0 %*% out1inv %*% ds2_train2$y

  tmpkname <- "kgaussiandotrelgradt1"
  tmp1 <- c(theta1[1], theta1[4:length(theta1)])
  nx <- ncol(ds2_train2mx)
  mtmp1 <- 0.5 * t(ds2_train2$y) %*% out1inv
  mtmp2 <- out1inv %*% ds2_train2$y
  relall <- rep(NA, nx)
  for (param2 in 1:nx) {
    #param2 is the rel parameter index, starting from 1
    param2 <- as.numeric(param2)
    cn_ptheta1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, tmpkname, tmp1, param2)

    rel_tmp <-  -0.5 * sum(diag(out1inv %*% cn_ptheta1)) + mtmp1 %*% cn_ptheta1 %*% mtmp2
    relall[param2] <- rel_tmp
  }
  return(c(gradbetainv, grad0, relall))
}


# gaussian dot kernels
loglike <- function(theta) {
  kname <- "gaussiandot"
  debug1 <- 0

  theta1 <- c(theta[1:2], 1, theta[3])
  betainv <- theta[4]
  param2 <- 1

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  logdet1 <- -0.5 * determinant(out1, logarithm = TRUE)$modulus[1]
  ip1 <- - 0.5 * t(ds2_train2$y) %*% out1inv %*% ds2_train2$y
  ret1 <- logdet1 + ip1
  flush.console()
  cat("   loglike: theta=", theta1, "betainv=",betainv, "ret1=", ret1, "logdet1=", logdet1, "ip1=", ip1, "\n")
  return(-1 * ret1)
}



# log likelihood using gaussian dot kernel with  relevance parameters for individual input features
loglikerel <- function(theta) {
  kname <- "gaussiandotrel"
  debug1 <- 0

  rel <- exp(theta)
  # loglikerel_param1
  theta1 <- c(loglikerel_param1[1], loglikerel_param1[3], loglikerel_param1[4], rel)

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  out1 <- out1 + loglikerel_betainv * diag(nrow(out1))
  # lambda = 1/100000
  result <- try(out1inv <- solve(out1), silent=TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  logdet1 <- -0.5 * determinant(out1, logarithm = TRUE)$modulus[1]
  ip1 <- - 0.5 * t(ds2_train2$y) %*% out1inv %*% ds2_train2$y
  ret1 <- logdet1 +ip1
  flush.console()
  cat("   loglike: theta=", theta1[1:3], "betainv=",betainv, "ret1=", ret1, "logdet1=", logdet1, "ip1=", ip1, "\n")
  cat("         theta[4:]=", theta1[4:length(theta1)], "\n")
  return(-1 * ret1)
}

grad_loglike <- function(theta) {
  debug1 <- 0
  kname <- "gaussiandot"

  theta1 <- c(theta[1:2], 1, theta[3])
  betainv <- theta[4]

  out1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, theta1, param2)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent=TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  # compute grad_theta0
  tmp1 <- c(1, theta1[1], 0, 0)
  cn_ptheta0 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, tmp1, param2)

  tmpkname <- "gaussiandotgradt1"
  tmp1 <- c(theta[1], theta[2], 0,0)
  cn_ptheta1 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, tmpkname, tmp1, param2)

  tmp1 <- c(0, 0, 0, 1)
  cn_ptheta3 <- tcrossprod_t(ds2_train2mx, ds2_train2mx, ifsym, debug1, kname, tmp1, param2)

  grad0 <- -0.5 * sum(diag(out1inv %*% cn_ptheta0)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% cn_ptheta0 %*% out1inv %*% ds2_train2$y
  grad1 <- -0.5 * sum(diag(out1inv %*% cn_ptheta1)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% cn_ptheta1 %*% out1inv %*% ds2_train2$y
  grad3 <- -0.5 * sum(diag(out1inv %*% cn_ptheta3)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% cn_ptheta3 %*% out1inv %*% ds2_train2$y
  gradbetainv <- -0.5 * sum(diag(out1inv)) + 0.5 * t(ds2_train2$y) %*% out1inv %*% out1inv %*% ds2_train2$y

  return(-1* c(grad0, grad1, grad3, gradbetainv))
}


