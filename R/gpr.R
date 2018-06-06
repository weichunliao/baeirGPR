#' @useDynLib baeirGPR
#' @importFrom Rcpp sourceCpp
# method = c("solve", "cg_direct", "cg_ichol")
gpr_train <- function(train_x, train_y, kparam, method = "solve",
                      ig_tol = 0.01, in_ncpu = -1) {
  obslist = c(1:nrow(train_x))
  #now, construct the full matrix bigK
  debug1 <- 0
  param2 <- 1
  nobs <- nrow(train_x)
  # cat("GPR training with", nobs, "data points\n")
  flush.console()

  if(method != "cg_direct_lm") {
    # cat("constructing bigK\n")
    flush.console()
    # cat(dim(train_x), 'zzzzzz\n')
    # aaa = rep(0,length(kparam$thetarel))
    # cat(debug1, kparam$kernelname, length(kparam$thetarel), '|', param2,'|', in_ncpu, "ppppp\n")
    # t1 <- system.time(bigK <- tcrossprod_t(train_x, train_x, 1, debug1, kparam$kernelname, aaa, param2, in_ncpu))

    t1 <- system.time(bigK <- tcrossprod_t(train_x, train_x, 1, debug1, kparam$kernelname, kparam$thetarel, param2, in_ncpu))
    # cat("bigK consumed time:\n")
    # print(t1)
    # return(9)
    flush.console()
    diagAddConst(bigK, kparam$betainv, 1)
  }

  # =====
  # now, construct bigMinv by doing incomplete choloskey decomposition,
  # and find the inverse of its cross product.

  # ifrunicol = TRUE
  # ifrunicol = FALSE

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
    t1 <- system.time(rvec <- as.matrix(train_y, ncol = 1) - as.matrix(kernelmdot(train_x, train_x, xvec, kparam$betainv, debug1, kparam$kernelname, kparam$thetarel, param2, in_ncpu), ncol = 1))
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
      bigKpvec <- as.matrix(kernelmdot(train_x, train_x, pvec, kparam$betainv, debug1, kparam$kernelname, kparam$thetarel, param2, in_ncpu), ncol=1)
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

  return(list(alpha = xvec, kparam = kparam, obslist = obslist))
  # return(list(alpha = xvec, kparam = kparam, obslist = obslist, obslist_y = train_y[obslist]))
}

gpr_predict <- function(testmx, trainmx, gprmodel, in_ncpu = -1) {
  debug1 <- 0
  param2 <- 1
  train_x <- trainmx[gprmodel$obslist,]

  t2 <- system.time(Ksmall2 <- tcrossprod_t(testmx, train_x, 0, debug1, gprmodel$kparam$kernelname, gprmodel$kparam$thetarel, param2, in_ncpu))
  # cat("Ksmall2 consumed time:\n")
  # print(t2)

  pred1 <- Ksmall2 %*% gprmodel$alpha
  return(pred1)
}


gpr_train_sr <- function(train_x, train_y, kparam, csize, in_ncpu) {
  # cat("gpr_sr trainer\n")

  # obslist <- c(1:csize)
  ndata <- nrow(train_x)

  try_inverse <- TRUE

  while (try_inverse) {

    obslist <- sample(ndata, csize)
    cat("gpr_train_sr: subset size=", csize, "\n")
    cat("training data size=", nrow(train_x), "\n")
    flush.console()

    subtrain <- train_x[obslist, ]
    debug1 <- 0
    param2 <- 1
    t1 <- system.time(out1 <- tcrossprod_t(subtrain, subtrain, 1, debug1, kparam$kernelname, kparam$thetarel, param2, in_ncpu))
    # cat("out1 consumed time:\n")
    # print(t1)

    debug1 <- 0
    sym2 <- 0
    # cat("compute kmn\n")
    flush.console()
    t1 <- system.time(kmn <- tcrossprod_t(subtrain, train_x, sym2, debug1, kparam$kernelname, kparam$thetarel, param2, in_ncpu))

    # cat("kmn consumed time:\n")
    # print(t1)
    flush.console()

    t1 <- system.time(kmn_cross <- tcrossprod(kmn))
    # cat("tcrossprod consumed time:\n")
    # print(t1)

    possibleError<- tryCatch({
      alpha <- solve((kmn_cross + kparam$betainv * out1), kmn %*% train_y)
    }, error = function(err) err
    )

    if(!inherits(possibleError, "error")) {
      try_inverse <- FALSE
      return(list(alpha = alpha, kparam = kparam, obslist = obslist))
    }
  }
}
