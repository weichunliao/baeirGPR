# cluster data and return a list of row indces of each cluster.
data_part <- function(x, partType="kmeans", nclus=10, iter.max = 100,
                      msize = 13000) {
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


# =======================
local_gpr_train <- function(train_x, train_y, kparam, obsClusters, ncpu = -1) {
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


local_gpr_predict <- function (test_x, lgpr_model, in_ncpu = -1) {

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
        t2 <- system.time(Ksmall2 <- tcrossprod_t(thistest, lgpr_model$train_x[obslist,], 0, debug1, lgpr_model$kparam$kernelname, lgpr_model$kparam$thetarel, param2, in_ncpu))
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

