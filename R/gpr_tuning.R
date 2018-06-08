#' Gaussian process regression kernel parameter tuning
#'
#' Tuning a gaussian process regression model's kernel parameter
#'
#' @param x Matrix; the features of tuning data set.
#' @param y Matrix; y.
#' @param kernel_name String; kernel name
#' @param ARD Boolean; set to TRUE to use ARD when tuning param;
#'   default value is TRUE.
#' @param init_betainv Numeric; initial value for kernel parameter, betainv.
#' @param init_theta Numeric vector; initail value for kernel parameter, theta.
#' @param optim_rbf_max Integer; max iteration when optimizing marginal log-likelihood.
#' @param optim_trace Integer;
#' @param optim_report Integer;
#' @param optim_ard_max Integer;
#' @param optim_ard_trace Integer;
#' @param optim_ard_report Integer;
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#'
#' @return  return a list having five objects:
#'   betainv
#'   thetarel
#'   nfeature
#'   kernelname
#'   ARD
#'
#' @export
gpr_tune <- function (x, y, kernelname = "rbf", ARD = TRUE,
                      init_betainv = NULL, init_theta = NULL,
                      optim_rbf_max = NULL, optim_trace = 1, optim_report = 5,
                      optim_ard_max = NULL, optim_ard_trace = 1,
                      optim_ard_report = 5, in_ncpu = -1) {
  if (is.null(optim_rbf_max)) {
    if (is.null(init_theta)) {
      optim_rbf_max = 100
    } else {
      optim_rbf_max = 20
    }
  }

  if (is.null(optim_ard_max)) {
    if (is.null(init_theta)) {
      optim_ard_max = 50
    } else {
      optim_ard_max = 20
    }
  }

  if(kernelname == "rbf") {
    kname <- "gaussiandotrel"
  } else {
    stop("gpr_turn: Unknown kernelname", kernelname, "\n")
  }
  cat("gpr_tune: optimizing marginal log-likelihood\n")
  # init_param = c(betainv, theta0, theta1)
  nfeature <- ncol(x)
  if (is.null(init_betainv)) {
    init_betainv <- 10
  }

  if (is.null(init_theta)) {
    init_theta <- c(2, 0.5)
    start_kparam <- c(init_betainv, init_theta)
    cat("    Using initial value [betainv theta0 theta1]:", start_kparam, "\n")
    flush.console()

    ans1 <- optim(start_kparam, loglike_gauss, grad_gauss_loglike, method = "L-BFGS-B",
                  lower = c(1e-6, 1e-6, 1e-6), upper = c(Inf, Inf, Inf),
                  control = list(maxit = optim_rbf_max, trace = optim_trace,
                                 REPORT = optim_report, fnscale = -1.0),
                  train_x = x, train_y = y, ncpu = in_ncpu)
    theta_for_stage1 <- c((ans1$par)[2:3])
    thetarel_noard <- c(ans1$par[2], 0, 0, rep(ans1$par[3], nfeature))
    betainv_noard <- ans1$par[1]
    cat("[NOARD] Optimal kernel parameter [theta0 theta1]=", theta_for_stage1, "\n")
    cat("[NOARD] Optimal kernel parameter betainv=", betainv_noard, "\n")

    cat("First stage optim (rbf, no ARD) convergence (0: success; 1: maxit reached) =", ans1$convergence, "msg =", ans1$message, "\n")
    flush.console()

    if (ARD == TRUE) {
      betainv <- betainv_noard
      rel <- c(betainv, theta_for_stage1[1], rep(theta_for_stage1[2], nfeature))
    }
  } else {
    init_theta0 <- init_theta[1]
    init_theta_i <- init_theta[c(4:length(init_theta))]
    if (ARD == FALSE) {
      start_kparam <- c(init_betainv, init_theta0, init_theta_i[1])
      cat("    Using initial value [betainv theta0 theta1]:", start_kparam, "\n")
      flush.console()



      ans1 <- optim(start_kparam[1:3], loglike_gauss, grad_gauss_loglike, method = "L-BFGS-B",
                    lower = c(1e-6, 1e-6, 1e-6), upper = c(Inf, Inf, Inf),
                    control = list(maxit = optim_rbf_max, trace = optim_trace,
                                   REPORT = optim_report, fnscale = -1.0),
                    train_x = x, train_y = y, ncpu = in_ncpu)
      theta_for_stage1 <- c((ans1$par)[2:3])
      thetarel_noard <- c(ans1$par[2], 0, 0, rep(ans1$par[3], nfeature))
      betainv_noard <- ans1$par[1]
      cat("[NOARD] Optimal kernel parameter [theta0 theta1]=", theta_for_stage1, "\n")
      cat("[NOARD] Optimal kernel parameter betainv=", betainv_noard, "\n")

      cat("First stage optim (rbf, no ARD) convergence (0: success; 1: maxit reached) =", ans1$convergence, "msg =", ans1$message, "\n")
      flush.console()
    } else {
      betainv <- init_betainv
      rel <- c(init_betainv, init_theta0, init_theta_i)
    }
  }

  if(ARD == FALSE) {
    outparam <- list(betainv = betainv_noard, thetarel = thetarel_noard, nfeature = nfeature,
                     kernelname = kname, ARD = FALSE)
  } else {
    cat("Running second stage optim (ARD)\n")
    flush.console()

    ans2 <- optim(rel, loglikerel2, grad_gauss_loglikerel2, method = "L-BFGS-B",
                  lower = rep(1e-6, length(rel)), upper = rep(Inf, length(rel)),
                  control = list(maxit = optim_ard_max, trace = optim_ard_trace,
                                 REPORT = optim_ard_report, fnscale = -1.0),
                  train_x = x, train_y = y, ncpu = in_ncpu)
    #
    cat("Second stage optim (RBF with ARD) convergence (0: success; 1: maxit reached) =", ans2$convergence, "msg =", ans2$message, "\n")
    flush.console()

    betainv <- ans2$par[1]
    thetarel <- c(ans2$par[2], 0, 0, ans2$par[3:length(ans2$par)])

    outparam <- list(betainv = betainv, thetarel = thetarel, nfeature = nfeature,
                     kernelname = kname, ARD = TRUE)

    cat("Optimal kernel parameter (ARD) betainv=", betainv, "\n")
    cat("Optimal kernel parameter (ARD) [theta0]=", ans2$par[2], "\n")
    cat("ARD param (head)=", head(thetarel[4:length(thetarel)], n = 10), "\n")
  }
  return(outparam)
}

# =========================
# tuning using local models
# gpr_tune_lm <- function(x, y, kernelname = "rbf", ARD = TRUE,
#                         init_param = c(10, 2, 0.5), clusters = NULL,
#                         optim_rbf_max = 100, optim_trace = 1,
#                         optim_report = 5, optim_ard_max = 20,
#                         optim_ard_trace = 1, optim_ard_report = 5,
#                         ncput = -1, in_ncpu = -1) {
#
#   if(kernelname == "rbf") {
#     kname <- "gaussiandotrel"
#   } else {
#     stop("gpr_turn: Unknown kernelname", kernelname, "\n")
#   }
#   cat("gpr_tune_lm: optimizing marginal log-likelihood\n")
#
#   param1 <- init_param
#   cat("    Using initial value [betainv theta0 theta1]:", param1, "\n")
#   flush.console()
#   nfeature <- ncol(x)
#
#   clus_x <- list()
#   clus_y <- list()
#   cc <- 0
#   for(alist in clusters) {
#     cc <- cc + 1
#     clus_x[[cc]] <- x[alist,]
#     clus_y[[cc]] <- y[alist,]
#   }
#
#
#   ans1 <- optim(param1, loglike_gauss_clus, grad_gauss_loglike_clus,
#                 method = "L-BFGS-B", lower = c(1e-6, 1e-6, 1e-6),
#                 upper = c(Inf, Inf, Inf), control = list(maxit = optim_rbf_max,
#                                                          trace = optim_trace,
#                                                          REPORT = optim_report,
#                                                          fnscale = -1.0),
#                 train_x_list = clus_x, train_y_list = clus_y, ncpu = in_ncpu)
#   param1 <- c((ans1$par)[2:3], 0, 0)
#   betainv <- (ans1$par)[1]
#
#   cat("Optimal kernel parameter [theta0 theta1]=", param1[1:2], "\n")
#   cat("Optimal kernel parameter betainv=", betainv, "\n")
#
#   cat("First stage optim (rbf, no ARD) convergence (0: success; 1: maxit reached)=", ans1$convergence, "msg=", ans1$message, "\n")
#   flush.console()
#
#   if(ARD == FALSE) {
#     thetarel <- c(param1[1], 0, 0, rep(param1[2], nfeature))
#     outparam <- list(betainv = betainv, thetarel = thetarel, nfeature = nfeature,
#                      kernelname = kname, ARD = FALSE)
#   } else {
#     cat("Running second stage optim (ARD)\n")
#     stop("not finished yet.")
#
#     thetarel_noard <- c(param1[1], 0, 0, rep(param1[2], nfeature))
#     betainv_noard <- betainv
#
#     flush.console()
#     # kname = "gaussiandotrel"
#
#     rel <- c(betainv, param1[1], rep(param1[2], nfeature))
#
#     ans2 <- optim(rel, loglikerel2, grad_gauss_loglikerel2, method = "L-BFGS-B",
#                   lower = rep(1e-6, length(rel)), upper = rep(Inf, length(rel)),
#                   control = list(maxit = optim_ard_max, trace = optim_ard_trace,
#                                  REPORT = optim_ard_report, fnscale = -1.0),
#                   ds2_train2mx = x, train_y = y, ncpu = in_ncpu)
#     #
#     cat("Second stage optim (RBF with ARD) convergence (0: success; 1: maxit reached)=", ans2$convergence, "msg=", ans2$message, "\n")
#     flush.console()
#
#     betainv <- ans2$par[1]
#     thetarel <- c(ans2$par[2], 0, 0, ans2$par[3:length(ans2$par)])
#     outparam <- list(betainv = betainv, thetarel = thetarel,
#                      nfeature = nfeature, kernelname = kname, ARD = TRUE,
#                      thetarel_noard = thetarel_noard,
#                      betainv_noard = betainv_noard)
#
#   }
#   return(outparam)
# }
# ########
#
# loglike_gauss_clus <- function(theta, train_x_list, train_y_list, ncpu) {
#   listn <- length(train_x_list)
#
#   sum1 <- 0.0
#   for (cc in 1:listn) {
#     sum1 <- sum1 + loglike_gauss(theta, train_x_list[[cc]], train_y_list[[cc]], ncpu)
#   }
#   return(sum1)
# }
#
# grad_gauss_loglike_clus <- function(theta, train_x_list, train_y_list, ncpu) {
#   listn <- length(train_x_list)
#
#   sum1 <- 0.0
#   for(cc in 1:listn) {
#     sum1 <- sum1 + grad_gauss_loglike(theta, train_x_list[[cc]], train_y_list[[cc]], ncpu)
#   }
#   return(sum1)
# }

# ===========================
# gaussian kernels
# theta [betainv theta1]
loglike_gauss <- function(theta, train_x, train_y, ncpu = -1) {
  kname = "gaussiandot"

  ifsym <- 1
  debug1 <- 0
  theta1 <- c(theta[2:3], 0, 0)
  betainv <- theta[1]
  param2 <- 1

  out1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, theta1, param2, ncpu)
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


grad_gauss_loglike <- function(theta, train_x, train_y, ncpu = -1) {
  debug1 <- 0
  ifsym <- 1
  kname <- "gaussiandot"

  theta1 <- c(theta[2:3], 0, 0)
  betainv <- theta[1]
  param2 <- 1

  out1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, theta1, param2, ncpu)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  # compute grad_theta0
  tmp1 <- c(1, theta1[2], 0, 0)
  cn_ptheta0 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, tmp1, param2, ncpu)

  tmpkname <- "gaussiandotgradt1"
  tmp1 <- c(theta1[1], theta1[2], 0,0)
  cn_ptheta1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, tmpkname, tmp1, param2, ncpu)

  grad0 <- -0.5 * sum(diag(out1inv %*% cn_ptheta0))  + 0.5 * t(train_y) %*% out1inv %*% cn_ptheta0 %*% out1inv %*% train_y
  grad1 <- -0.5 * sum(diag(out1inv %*% cn_ptheta1))  + 0.5 * t(train_y) %*% out1inv %*% cn_ptheta1 %*% out1inv %*% train_y
  gradbetainv <- -0.5 * sum(diag(out1inv))  + 0.5 * t(train_y) %*% out1inv %*% out1inv %*% train_y

  return(c(gradbetainv, grad0, grad1))
}


# marginal log-likelihood using gaussian kernel with individual relevance parameters for input features
# theta = [betainv rel]
loglikerel2 <- function(theta, train_x, train_y, ncpu) {
  debug1 <- 0
  ifsym <- 1
  param2 <- 1

  tmp1 <- theta
  kname <- "gaussiandotrel"

  theta1 <- c(tmp1[2], 0, 0, tmp1[3:length(tmp1)])
  betainv <- tmp1[1]

  out1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, theta1, param2, ncpu)
  out1 <- out1 + betainv * diag(nrow(out1))
  # lambda = 1/100000
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }


  logdet1 <- -0.5 * determinant(out1, logarithm = TRUE)$modulus[1]
  ip1 <- - 0.5 * t(train_y) %*% out1inv %*% train_y
  ret1 <- logdet1 + ip1
  flush.console()
  # cat("   loglike: theta=", theta1[1:3], "betainv=",betainv, "ret1=", ret1, "logdet1=", logdet1, "ip1=", ip1, "\n")
  # cat("         theta[4:]=", theta1[4:length(theta1)], "\n")
  return(ret1)
}


grad_gauss_loglikerel2 <- function(theta, train_x, train_y, ncpu) {
  debug1 <- 0
  kname <- "gaussiandotrel"
  ifsym <- 1
  param2 <- 1

  theta1 <- c(theta[2], 0, 0, theta[3:length(theta)])
  betainv <- theta[1]

  out1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, theta1, param2, ncpu)
  # lambda = 1/100000
  out1 <- out1 + betainv * diag(nrow(out1))
  result <- try(out1inv <- solve(out1), silent = TRUE)
  if (class(result) == "try-error") {
    return(NaN)
  }

  gradbetainv <- -0.5 * sum(diag(out1inv)) + 0.5 * t(train_y) %*% out1inv %*% out1inv %*% train_y

  # compute grad_theta0
  tmp1 <- c(1, 0, 0, theta1[4:length(theta1)])
  cn_ptheta0 <- tcrossprod_t(train_x, train_x, ifsym, debug1, kname, tmp1, param2, ncpu)
  grad0 <- -0.5 * sum(diag(out1inv %*% cn_ptheta0)) + 0.5 * t(train_y) %*% out1inv %*% cn_ptheta0 %*% out1inv %*% train_y

  tmpkname <- "kgaussiandotrelgradt1"
  tmp1 <- c(theta1[1], theta1[4:length(theta1)])
  n_feature <- ncol(train_x)
  mtmp1 <- 0.5 * t(train_y) %*% out1inv
  mtmp2 <- out1inv %*% train_y
  relall <- rep(NA, n_feature)
  for (param2 in 1:n_feature) {
    #param2 is the rel parameter index, starting from 1
    param2 <- as.numeric(param2)
    cn_ptheta1 <- tcrossprod_t(train_x, train_x, ifsym, debug1, tmpkname, tmp1, param2, ncpu)

    rel_tmp <-  -0.5 * sum(diag(out1inv %*% cn_ptheta1)) + mtmp1 %*% cn_ptheta1 %*% mtmp2
    relall[param2] <- rel_tmp
  }
  return(c(gradbetainv, grad0, relall))
}
