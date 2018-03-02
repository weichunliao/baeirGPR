gbm_train <- function(train_x, train_y, test_x, test_y, pred_method = "1",
                      n_model = 500, batch_size = 1000, lr = 0.1, tune_param = FALSE, tune_size = NULL,
                      update_kparam_tiems = 50, update_col_sample = 50,
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
      # pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data
      #
      # if(iter_id > 1){
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      # } else {
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      # }
      # all_test_rmse[iter_id] <- rmse_test
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
          kparam <- gpr_tune(train_x[tune_ind,], adj_y[tune_ind])
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
      # pred_test_y <- gpr_predict(test_x, train_x[train_ind,], temp_model) ##testing data
      #
      # if(iter_id > 1){
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      # } else {
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      # }
      # all_test_rmse[iter_id] <- rmse_test
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
          kparam <- gpr_tune(train_x[tune_ind,], adj_y[tune_ind])
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
        kparam <- gpr_tune(train_x[tune_ind,col_sampling_idx], adj_y[tune_ind], ARD = FALSE)
      }
      cat("Now, running for iteration", iter_id, "\n")

      train_ind <- sample(n_data, batch_size)

      temp_model <- gpr_train(train_x[train_ind,col_sampling_idx], adj_y[train_ind], kparam)
      temp_model$sub_sample_idx <- train_ind
      temp_model$lr <- lr
      temp_model$col_sampling_idx <- col_sampling_idx
      all_gpr_models[[iter_id]] <- temp_model
      # /////////
      # pred_test_y <- gpr_predict(test_x[,col_sampling_idx], train_x[train_ind,col_sampling_idx], temp_model)
      # if (iter_id > 1) {
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y*lr)^2))
      # } else {
      #   rmse_test <- sqrt(mean((adj_test_y - pred_test_y)^2))
      # }
      # all_test_rmse[iter_id] <- rmse_test
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
  return( list(models = all_gpr_models, pred_method = pred_method, train_rmse = all_train_rmse, test_rmse = all_test_rmse) )
}

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

