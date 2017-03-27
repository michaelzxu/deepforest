deepforest <- function(x, y, x_val = NULL, y_val =NULL, nlayer = 5, nmeta = 4,
                       nclass = NULL, nfold = 5, index = NULL,
                       eval_metric = NULL, missing = NA, metaparam = NULL,
                       ntree = 10, subsample = 0.632, colsample = 0.2,
                       printby = "fold", colsample_bylayer = 0.5,
                       ...) {
    if (is.null(index)) {
        index <- caret::createFolds(y, k = nfold, list = TRUE)
    } else {
        if (class(index) != "list") stop("index must be a list")
        nfold <- length(index)
    }

    if (is.null(metaparam)) {
        metaparam <- rep(list(model = "xgbforest",
                              args = list(num_parallel_tree = 500), nmeta))
    }

    if (is.null(nclass)) {
        if (length(unique(y)) < 10) {
            nclass <- length(unique(y))
        } else {
            nclass <- 1
        }
    }

    if (is.null(eval_metric)) {
        if (nclass == 1) {
            eval_metric <- Metrics::mse
        } else if (nclass == 2) {
            eval_metric <- Metrics::auc
        } else if (nclass > 2) {
            eval_metric <- Metrics::logLoss
        }
    }

    x <- as.matrix(x)
    if (is.null(colnames(x))) {
        colnames(x) <- paste0("X", 1:ncol(x))
    }

    if (is.null(x_val)) {
        x_val <- x
        y_val <- y
    } else {
        x_val <- as.matrix(x_val)
        if (is.null(colnames(x_val))) {
            colnames(x_val) <- paste0("X", 1:ncol(x_val))
        }
        if (is.null(y_val)) {
            stop("y_val not provided when x_val is not NULL")
        }
    }

    orig_x <- x
    orig_x_val <- x_val

    .lcols <- sample(ncol(x))[1:floor(ncol(x) * colsample_bylayer)]
    x <- x[ , .lcols]
    x_val <- x_val[ , .lcols]

    .strong <- vector("list", nlayer)
    .strongeval <- numeric(nlayer)
    .model <- rep(list(rep(list(vector("list", nfold)), nmeta)), nlayer)
    .eval <- array(dim = c(nlayer, nmeta, nfold))

    for (.l in 1:nlayer) {
        .colnm <- paste0("L", .l, "_M", rep(1:nmeta, each =
                                                ifelse(nclass == 2, 1, nclass)),
                         "_C", rep(1:ifelse(nclass == 2, 1, nclass), nmeta))
        .x.l <- matrix(nrow = nrow(x), ncol = nmeta * ifelse(nclass == 2, 1,
                                                             nclass),
                       dimnames = list(NULL, .colnm))
        .x_val.l <- matrix(nrow = nrow(x_val), ncol = nmeta *
                               ifelse(nclass == 2, 1, nclass),
                           dimnames = list(NULL, .colnm))
        .csbt <- ifelse(colsample * ncol(x) <= 1, 1 / ncol(x), colsample)
        .mxd <- 99999
        .mxl <- 99999
        .subs <- subsample
        .ntree <- ntree

        for (.m in 1:nmeta) {
            x_val.m <- xgboost::xgb.DMatrix(data = x_val,label = y_val,
                                            missing = missing)
            if (printby == "meta") {
                cat(paste0("Layer ", .l, ", Meta ", .m))
            }
            for (.f in 1:nfold) {
                if (printby == "fold") {
                    cat(paste0("Layer ", .l, ", Meta ", .m, ", Fold ", .f))
                }
                x.f <- xgboost::xgb.DMatrix(data = x[-index[[.f]], ],
                                            label = y[-index[[.f]]],
                                            missing = missing)
                if (nclass == 1) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1,
                        params = list(colsample_bytree = .csbt,
                        subsample = .subs, eta = 1, max_leaves = .mxl,
                        num_parallel_tree = .ntree, max_depth = .mxd),
                        watchlist = list(val = x_val.m),
                        objective = "reg:linear", verbose = FALSE, ...)
                    .x.l[index[[.f]], .m] <- predict(.model[[.l]][[.m]][[.f]],
                                                     x[index[[.f]], ])
                } else if (nclass == 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1,
                        params = list(colsample_bytree = .csbt,
                        subsample = .subs, eta = 1, max_leaves = .mxl,
                        num_parallel_tree = .ntree, max_depth = .mxd),
                        watchlist = list(val = x_val.m),
                        objective = "binary:logistic", verbose = FALSE,
                        ...)
                    .x.l[index[[.f]], .m] <- predict(.model[[.l]][[.m]][[.f]],
                                                     x[index[[.f]], ])
                } else if (nclass > 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1,
                        params = list(colsample_bytree = .csbt,
                        subsample = .subs, eta = 1, max_leaves = .mxl,
                        num_parallel_tree = .ntree, max_depth = .mxd),
                        num_class = nclass, objective = "multi:softprob",
                        watchlist = list(val = x_val.m),
                        verbose = FALSE, ...)
                    .x.l[index[[.f]], ((.m - 1) * nclass + 1):(.m * nclass)] <-
                        predict(.model[[.l]][[.m]][[.f]], x[index[[.f]], ],
                                reshape = TRUE)
                } else {
                    stop("Invalid nclass")
                }
                if (nclass > 2) {
                    .eval[.l, .m, .f] <- eval_metric(y_val,apply(
                        predict(.model[[.l]][[.m]][[.f]], x_val.m, reshape = TRUE),
                        1, which.max))
                } else {
                    .eval[.l, .m, .f] <-
                        eval_metric(y_val, predict(.model[[.l]][[.m]][[.f]],
                                                   x_val.m))
                }
                if (printby == "fold") {
                    cat(paste0(", Eval ", .eval[.l, .m, .f], "\n"))
                }
                gc()
            }
            if (printby %in% c("fold","meta")) {
                cat(paste0(", Eval ", mean(.eval[.l, .m, ]), "\n"))
            }
            if (nclass <= 2) {
                .x_val.l[ , .m] <-
                    apply(simplify2array(lapply(.model[[.l]][[.m]], predict,
                                                newdata = x_val.m)),1,mean)
            } else if (nclass > 2) {
                .x_val.l[ , ((.m - 1) * nclass + 1):(.m * nclass)] <-
                    apply(simplify2array(
                        lapply(.model[[.l]][[.m]], predict, newdata = x_val.m)),
                        1:nmeta, mean)
            }
            gc()
        }

        if (.l == 1) {
            .strong[[.l]] <-
                xgboost::xgb.train(data = xgboost::xgb.DMatrix(data = .x.l,
                                                               label = y,
                                                               missing = missing),
                                   nrounds = 20, objective = "binary:logistic",
                                   params = list(eta = 0.3, subsample = 0.5,
                                                 colsample_bytree = max(1/ncol(x.tmp),0.5),
                                                 max_depth = 5),
                                   verbose = FALSE)
            .strongeval[.l] <- eval_metric(y, predict(.strong[[.l]], .x.l))
            tmpscore <- eval_metric(y_val, predict(.strong[[.l]], .x_val.l))
        } else {
            .strong[[.l]] <-
                xgboost::xgb.train(data = xgboost::xgb.DMatrix(data = cbind(x.tmp, .x.l),
                                                               label = y,
                                                               missing = missing),
                                   nrounds = 20, objective = "binary:logistic",
                                   params = list(eta = 0.3, subsample = 0.5,
                                                 colsample_bytree = max(1/ncol(x),0.5),
                                                 max_depth = 5),
                                   verbose = FALSE)
            .strongeval[.l] <- eval_metric(y, predict(.strong[[.l]], cbind(x.tmp, .x.l)))
            tmpscore <- eval_metric(y_val, predict(.strong[[.l]], cbind(x_val.tmp, .x_val.l)))
        }

        .lcols <- sample(ncol(orig_x))[1:floor(ncol(orig_x) * colsample_bylayer)]
        if (.l == 1) {
            x.tmp <- .x.l
            x_val.tmp <- .x_val.l
        } else {
            if (.strongeval[.l] > .strongeval[.l - 1]) {
                x.tmp <- cbind(x.tmp, .x.l)
                x_val.tmp <- cbind(x_val.tmp, .x_val.l)
            }
        }
        x <- cbind(x.tmp, orig_x[ , .lcols])
        x_val <- cbind(x_val.tmp, orig_x_val[ , .lcols])

        # .lcols <- sample(ncol(orig_x))[1:floor(ncol(orig_x) * colsample_bylayer)]
        # x <- cbind(.x.l, orig_x[ , .lcols])
        # x_val <- cbind(.x_val.l, orig_x_val[ , .lcols])

        cat(paste0("Layer ", .l, " average: ", mean(.eval[.l, , ]), "\n"))

        cat(paste0("Layer ", .l, " expert: ", tmpscore, "\n"))
        gc()
    }
    return(structure(list(model = .model, eval = .eval), class = "deepforest"))
}

mod <- deepforest(credc.train,credc[intrain,"Class"],
                  credc.test,credc[-intrain,"Class"],
                  nmeta=4, nlayer=10, nfold = 4,
                  ntree = 1000, colsample = 0.6, subsample = 0.1, printby = "meta",
                  colsample_bylayer = 0.2)


a<-xgboost::xgb.DMatrix(data=as.matrix(iris[,1:4],label=as.numeric(iris[,5]=="setosa")))
mod<-xgboost::xgb.train(data= a, nrounds=1,num_parallel_tree = 3,colsample_bytree=0.25,objective="binary:logistic")
a<-xgboost::xgb.DMatrix(data=as.matrix(iris[,1:4],label=as.integer(iris[,5])))
mod<-xgboost::xgb.train(data= a, nrounds=1,num_parallel_tree = 3,colsample_bytree=0.25,objective="multi:softprob",num_class=3)
predict(mod, a, predleaf=T)
predict(mod, a, reshape=TRUE)
