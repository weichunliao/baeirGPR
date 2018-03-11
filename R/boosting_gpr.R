#' Gradient boosting machine based on GPR
#'
#' Training a gradient boosting machine
#'
#' @param train_x Matrix; the features of training data set.
#' @param train_y Matrix; y.
#' @param pred_method String; Set the model training approach.
#'   1: random row sampling after all training data have been used.
#'   2: random row sampling.
#'   3: row sampling plus col sampling.
#' @param tune_param Boolean; Set to TRUE to tune parameters of kernel function
#'   default value is TRUE.
#' @param n_model Non-negative integer; the number of submodel in gbm.
#' @param batch_size Non-negative integer; batch size for each iteration.
#' @param lr Numeric between 0-1; learning rate.
#' @param tune_param Boolean; Set to TRUE to tune parameters of kernel function,
#'   default value for pred_method 1 & 2 is TRUE.
#'   default value for pred_method 3 is FALSE.
#' @param tune_size Non-negative integer; size of tuning data set.
#' @param update_kparam_times Non-negative integer; time to update kernel parameter(for method 1/2).
#' @param update_col_sample Non-negative integer; time to update kernel parameter(for method 3).
#' @param kname String; the name of kernel; default value is 'gaussiandotrel'.
#' @param ktheta Numeric vector; store kernel parameter;
#'   should be provided when tune_param is FALSE.
#' @param kbetainv Numeric; store kernel parameter betainv;
#'   shuld be provided when tune_param is FALSE.
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#'
#' @return  return a list having four objects:
#'   models
#'   pred_method
#'   train_rmse
#'   test_rmse
#'
#' @export
gbm_train <- function(train_x, train_y, test_x, test_y, pred_method = "1",
                      n_model = 500, batch_size = 1000, lr = 0.1,
                      tune_param = FALSE, tune_size = NULL,
                      update_kparam_tiems = 50, update_col_sample = 50,
                      kname = "gaussiandotrel", ktheta = NULL,
                      kbetainv = NULL, ncpu = -1) {
# gbm_train <- function(train_x, train_y, pred_method = "1",
#                       n_model = 500, batch_size = 1000, lr = 0.1,
#                       tune_param = FALSE, tune_size = NULL,
#                       update_kparam_tiems = 50, update_col_sample = 50,
#                       kname = "gaussiandotrel", ktheta = NULL,
#                       kbetainv = NULL, ncpu = -1) {
  if (is.null(ktheta) | is.null(kbetainv)) {
    cat("[ERROR] miss kernel parameter.")
    return(-1)
  }

  if (is.null(tune_param)) {
    if (pred_method == "3") {
      tune_param <- FALSE
    } else {
      tune_param <- TRUE
    }
  }
  kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
  all_gpr_models <- vector("list", n_model)

  n_data <- nrow(train_x)
  all_test_rmse <- rep(NA, n_model)
  all_train_rmse <- rep(NA, n_model)

  adj_y <- train_y
  adj_test_y <- test_y
  pred_train_y <- rep(0, n_data)
  pred_test_y <- rep(0, length(test_y))

  if (is.null(tune_size)) {
    tune_size <- batch_size
  }

  if (pred_method == "1") {
    # cat("pred_method == 1\n")
    max_iter <- floor(n_data/batch_size)
    used_train <- rep(FALSE, nrow(train_x))

    for(iter_id in 1:n_model) {
      cat("Now, running for iteration", iter_id, "\n")
      if(iter_id <= max_iter) {
        ind1 <- (iter_id-1)*batch_size+1
        ind2 <- iter_id*batch_size
        used_train[ind1:ind2] <- TRUE
        train_ind <- ind1:ind2
      } else {
        train_ind <- sample(n_data, batch_size)
      }

      temp_model <- gpr_train(train_x[train_ind,], adj_y[train_ind], kparam)
      temp_model$sub_sample_idx <- train_ind
      temp_model$lr <- lr
      all_gpr_models[[iter_id]] <- temp_model
      # ///////
      pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data

      if(iter_id > 1){
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test
      # //////////
      # cat(" (GPR noard) rmse (test); ", "iter_id=", iter_id, "->", rmse_test, "\n")
      pred_train_y <- gpr_predict(train_x, train_x[train_ind,], temp_model) ##training data
      rmse_train <- sqrt(mean((adj_y - pred_train_y)^2))
      all_train_rmse[iter_id] <- rmse_train
      # cat(" (GPR noard) rmse (train); ", "iter_id=", iter_id, "->", rmse_train, "\n")

      if(iter_id > 1){
        adj_y <- adj_y - pred_train_y * lr
        adj_test_y <- adj_test_y - pred_test_y * lr
      } else {
        adj_y <- adj_y - pred_train_y
        adj_test_y <- adj_test_y - pred_test_y
      }

      if (tune_param == TRUE) {
        if(iter_id %% update_kparam_tiems == 0) {
          cat("update theta......\n")
          # tune_size <- batch_size
          tune_ind <- sample(n_data, tune_size)
          kparam <- gpr_tune(train_x[tune_ind,], adj_y[tune_ind], ARD = tune_param, init_betainv = kparam$betainv, init_theta = kparam$thetarel)
        }
      }
    }
  } else if (pred_method == "2") {
    # cat("pred_method == 2\n")
    max_iter <- floor(n_data/batch_size)

    for(iter_id in 1:n_model) {
      kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
      cat("Now, running for iteration", iter_id, "\n")

      train_ind <- sample(n_data, batch_size)

      temp_model <- gpr_train(train_x[train_ind,], adj_y[train_ind], kparam)
      temp_model$sub_sample_idx <- train_ind
      temp_model$lr <- lr
      all_gpr_models[[iter_id]] <- temp_model
      # //////////
      pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data

      if(iter_id > 1){
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test
      # //////////
      # cat(" (GPR noard) rmse (test); ", "iter_id=", iter_id, "->", rmse_test, "\n")
      pred_train_y <- gpr_predict(train_x, train_x[train_ind,], temp_model) ##training data
      rmse_train <- sqrt(mean((adj_y - pred_train_y)^2))
      all_train_rmse[iter_id] <- rmse_train
      # cat(" (GPR noard) rmse (train); ", "iter_id=", iter_id, "->", rmse_train, "\n")

      if(iter_id > 1){
        adj_y <- adj_y - pred_train_y * lr
        adj_test_y <- adj_test_y - pred_test_y * lr
      } else {
        adj_y <- adj_y - pred_train_y
        adj_test_y <- adj_test_y - pred_test_y
      }
      if (tune_param == TRUE) {
        if(iter_id %% update_kparam_tiems == 0) {
          cat("update theta......\n")
          tune_ind <- sample(n_data, tune_size)
          kparam <- gpr_tune(train_x[tune_ind,], adj_y[tune_ind], ARD = tune_param, init_betainv = kparam$betainv, init_theta = kparam$thetarel)
        }
      }
    }
  } else if (pred_method == "3") {
    # cat("pred_method == 3\n")
    origin_kparam <- kparam

    n_feature <- ncol(train_x)
    selected_n_feature <- floor(sqrt(n_feature))
    for(iter_id in 1:n_model) {
      if(iter_id %% update_col_sample == 1) {
        # cat("col sampling...\n")
        col_sampling_idx <- sort(sample(c(1:n_feature), selected_n_feature))
        # col_sampling_train_x <- train_x[,col_sampling_idx]
        # tune_size <- 100
        tune_ind <- sample(n_data, tune_size)
        kparam <- gpr_tune(train_x[tune_ind,col_sampling_idx], adj_y[tune_ind], ARD = tune_param, init_betainv = kparam$betainv, init_theta = kparam$thetarel)
      }
      cat("Now, running for iteration", iter_id, "\n")

      train_ind <- sample(n_data, batch_size)

      temp_model <- gpr_train(train_x[train_ind,col_sampling_idx], adj_y[train_ind], kparam)
      temp_model$sub_sample_idx <- train_ind
      temp_model$lr <- lr
      temp_model$col_sampling_idx <- col_sampling_idx
      all_gpr_models[[iter_id]] <- temp_model
      # /////////
      pred_test_y <- gpr_predict(test_x[,col_sampling_idx], train_x[train_ind,col_sampling_idx], temp_model)
      if (iter_id > 1) {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test
      # /////////
      pred_train_y <- gpr_predict(train_x[,col_sampling_idx], train_x[train_ind,col_sampling_idx], temp_model)
      rmse_train <- sqrt(mean((adj_y - pred_train_y)^2))
      all_train_rmse[iter_id] <- rmse_train

      if(iter_id > 1){
        adj_y <- adj_y - pred_train_y * lr
        adj_test_y <- adj_test_y - pred_test_y * lr
      } else {
        adj_y <- adj_y - pred_train_y
        adj_test_y <- adj_test_y - pred_test_y
      }
    }
  }
  # plot(all_test_rmse[1:n_model], type = 'l')
  # plot(all_train_rmse[1:n_model], type = 'l')
  # cat("train_rmse:", all_train_rmse, "\n")
  # cat("test_rmse:", all_test_rmse, "\n")
  # return( list(models = all_gpr_models, pred_method = pred_method, train_rmse = all_train_rmse) )
  return( list(models = all_gpr_models, pred_method = pred_method, train_rmse = all_train_rmse, test_rmse = all_test_rmse) )
}

#' Gradient boosting machine based on GPR
#'
#' GBM prediction
#'
#' @param testmx Matrix; the features of testing dateset.
#' @param trainmx Matrix; the features of training data set.
#' @param gbm_model List; the output of gbm_train(), containing three objects:
#'   models
#'   pred_method
#'   train_rmse
#' @param ncpu Integer; the number of thread to be used;
#'   set to -1 to use all threads; default value is -1.
#'
#' @return List;
#'   prediction: the final prediction on testing dataset.
#'   test_rmse: numeric vector; list of rmse during testing.
#'
#' @export
gbm_fit <- function(testmx, test_y, trainmx, gbm_model, ncpu = -1){
  models <- gbm_model$models
  pred_method <- gbm_model$pred_method
  n_model <- length(models)
  accumulate_test_y <- rep(0, nrow(testmx))
  all_test_rmse <- rep(NA, n_model)

  for (iter_id in 1:n_model) {
    cat("the i-th iteration of prediction:", iter_id, "\n")
    if ((pred_method == "1") | (pred_method == "2")) {
      pred_test_y = gpr_predict(testmx, trainmx[models[[iter_id]]$sub_sample_idx,], models[[iter_id]])
    } else if (pred_method == "3") {
      # gpr_predict(test_x[,col_sampling_idx], train_x[train_ind,col_sampling_idx], temp_model)
      pred_test_y = gpr_predict(testmx[, models[[iter_id]]$col_sampling_idx], trainmx[models[[iter_id]]$sub_sample_idx, models[[iter_id]]$col_sampling_idx], models[[iter_id]])
    }
    if (iter_id > 1) {
      accumulate_test_y <- accumulate_test_y + (pred_test_y * models[[iter_id]]$lr)
    } else {
      accumulate_test_y <- pred_test_y
    }
    rmse_test <- sqrt(mean((test_y - accumulate_test_y)^2))
    all_test_rmse[[iter_id]] <- rmse_test
    cat("iter_id - ", iter_id, ":", rmse_test, "\n")
  }
  plot(all_test_rmse[1:n_model], type = 'l')
  # cat("test_rmse:", all_test_rmse, "\n")
  return(list(prediction=accumulate_test_y, test_rmse = all_test_rmse))
}
