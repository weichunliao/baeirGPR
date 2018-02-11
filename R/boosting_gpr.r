gbm_train <- function(train_x, train_y, test_x, test_y, pred_method = "1",
                      n_model = 50, batch_size = 10, lr = 0.1, tune_param = FALSE,
                      kname = "gaussiandotrel", ktheta = NULL,
                      kbetainv = NULL, ncpu = -1) {
  kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
  all_gpr_models <- vector("list", n_model)

  n_data <- nrow(train_x)
  all_test_rmse <- rep(NA, n_model)
  all_train_rmse <- rep(NA, n_model)

  adj_y <- train_y
  adj_test_y <- test_y
  pred_train_y <- rep(0, n_data)
  pred_test_y <- rep(0, length(test_y))

  if (pred_method == "1") {
    cat("pred_method == 1\n")
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
      pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data

      if(iter_id > 1){
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test
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
        if(iter_id %% 10 == 0) {
          cat("update theta......\n")
          train_ind <- sample(n_data, batch_size)
          kparam <- gpr_tune(train_x[train_ind,], adj_y[train_ind])
        }
      }
    }
  } else if (pred_method == "2") {
    cat("pred_method == 2\n")
    max_iter <- floor(n_data/batch_size)

    for(iter_id in 1:n_model) {
      kparam <- list(betainv = kbetainv, thetarel = ktheta, kernelname = kname)
      cat("Now, running for iteration", iter_id, "\n")

      train_ind <- sample(n_data, batch_size)

      temp_model <- gpr_train(train_x[train_ind,], adj_y[train_ind], kparam)
      temp_model$sub_sample_idx <- train_ind
      temp_model$lr <- lr
      all_gpr_models[[iter_id]] <- temp_model
      pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data

      if(iter_id > 1){
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test
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
    }
  } else if (pred_method == "3") {
    cat("pred_method == 3\n")
    origin_kparam <- kparam
    max_iter <- floor(n_data/batch_size)

    used_train <- rep(FALSE, nrow(train_x))

    n_theta <- length(origin_kparam$thetarel)
    n_feature <- n_theta-3
    selected_n_feature <- floor(sqrt(n_feature))
    for(iter_id in 1:n_model) {
      if(iter_id > 50) {
        selected_feature_idx <- sample(c(4:n_feature), selected_n_feature)
        adj_feature_theta <- origin_kparam$thetarel
        adj_feature_theta[-selected_feature_idx] <- 0

        kparam$thetarel <- adj_feature_theta
      }
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
      pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model)
      if (iter_id > 1) {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      } else {
        rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      }
      all_test_rmse[iter_id] <- rmse_test

      pred_train_y <- gpr_predict(train_x, train_x[train_ind,], temp_model)
      rmse_train <- sqrt(mean((adj_y - pred_train_y)^2))
      all_train_rmse[iter_id] <- rmse_train

      if(iter_id > 1){
        adj_y <- adj_y - pred_train_y * lr
        adj_test_y <- adj_test_y - pred_test_y * lr
      } else {
        adj_y <- adj_y - pred_train_y
        adj_test_y <- adj_test_y - pred_test_y
      }

      if (tune_param == TRUE) {
        if(iter_id %% 10 == 0) {
          cat("update theta......\n")
          train_ind <- sample(n_data, batch_size)
          kparam <- gpr_tune(train_x[train_ind,], adj_y[train_ind])
        }
      }
    }
  }
  plot(all_test_rmse[1:n_model], type = 'l')
  # plot(all_train_rmse[1:n_model], type = 'l')
  cat("train_rmse:", all_train_rmse, "\n")
  cat("test_rmse:", all_test_rmse, "\n")
  return(all_gpr_models)
}

gbm_fit <- function(testmx, test_y, trainmx, gbm_model, ncpu = -1){
  n_model <- length(gbm_model)
  accumulate_test_y <- rep(0, nrow(testmx))
  all_test_rmse <- rep(NA, n_model)
  for (iter_id in 1:n_model) {
    cat("the i-th iteration of prediction:", iter_id, "\n")
    pred_test_y = gpr_predict(testmx, trainmx[gbm_model[[iter_id]]$sub_sample_idx,], gbm_model[[iter_id]]) ##testing data
    if (iter_id > 1) {
      accumulate_test_y <- accumulate_test_y + (pred_test_y * gbm_model[[iter_id]]$lr)
    } else {
      accumulate_test_y <- pred_test_y
    }
    rmse_test <- sqrt(mean((test_y - accumulate_test_y)^2))
    all_test_rmse[[iter_id]] <- rmse_test
    cat("iter_id - ", iter_id, ":", rmse_test, "\n")
  }
  cat("test_rmse:", all_test_rmse, "\n")
  return(accumulate_test_y)
}

