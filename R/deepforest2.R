deepforest <- function(x, y, x_val = NULL, y_val =NULL, nlayer = 5, nmeta = 4,
                       nclass = NULL, nfold = 5, index = NULL,
                       eval_metric = NULL, missing = NA, metaparam = NULL,
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

    .model <- rep(list(rep(list(vector("list", nfold)), nmeta)), nlayer)
    .eval <- array(dim = c(nlayer, nmeta, nfold))

    for (.l in 1:nlayer) {
        .colnm <- paste0("L", .l, "_M", rep(1:nmeta, each = nclass),
                         "_C", rep(1:nclass, nmeta))
        .x.l <- matrix(nrow = nrow(x), ncol = nmeta * nclass,
                       dimnames = list(NULL, .colnm))
        .x_val.l <- matrix(nrow = nrow(x_val), ncol = nmeta * nclass,
                           dimnames = list(NULL, .colnm))
        for (.m in 1:nmeta) {
            x_val.m <- xgboost::xgb.DMatrix(data = x_val,label = y_val,
                                            missing = missing)
            for (.f in 1:nfold) {
                cat(paste0("Layer ", .l, ", Meta ", .m, ", Fold ", .f))
                x.f <- xgboost::xgb.DMatrix(data = x[-index[[.f]], ],
                                            label = y[-index[[.f]]],
                                            missing = missing)
                if (nclass == 1) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1, colsample_bytree=0.1,
                        num_parallel_tree = 1000, max_depth=5,
                        watchlist = list(val = x_val.m),
                        objective = "reg:linear", verbose = FALSE, ... = ...)
                } else if (nclass == 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1, colsample_bytree=0.1,
                        num_parallel_tree = 1000, max_depth=5,
                        watchlist = list(val = x_val.m),
                        objective = "binary:logistic", verbose = FALSE,
                        ... = ...)
                } else if (nclass > 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = 1, colsample_bytree=0.1,
                        num_parallel_tree = 1000, max_depth=5,
                        num_class = nclass, objective = "multi:softprob",
                        watchlist = list(val = x_val.m),
                        verbose = FALSE, ... = ...)
                } else {
                    stop("Invalid nclass")
                }
                .x.l[index[[.f]], ((.m - 1)*nclass + 1):(.m * nclass)] <-
                    predict(.model[[.l]][[.m]][[.f]], x[index[[.f]], ],
                            reshape = TRUE)
                if (nclass > 2) {
                    .eval[.l, .m, .f] <- eval_metric(y_val,apply(
                        predict(.model[[.l]][[.m]][[.f]], x_val.m, reshape = TRUE),
                        1, which.max))
                } else {
                    .eval[.l, .m, .f] <-
                        eval_metric(y_val, predict(.model[[.l]][[.m]][[.f]],
                                                   x_val.m))
                }

                cat(paste0(", Eval ", .eval[.l, .m, .f], "\n"))
            }
            .x_val.l[ , ((.m - 1) * nclass + 1):(.m * nclass)] <-
                apply(simplify2array(lapply(.model[[.l]][[.m]], predict,
                                            newdata = x_val.m, reshape = TRUE)),
                      1:nmeta, mean)
        }
        x <- .x.l
        x_val <- .x_val.l
    }
    return(structure(list(model = .model, eval = .eval)))
}

mod<-deepforest(credc.train,credc[intrain,"Class"],credc.test,credc[-intrain,"Class"])


mod<-xgboost::xgb.train(data= a, nrounds=1,num_parallel_tree = 3,colsample_bytree=0.25,objective="multi:softprob",num_class=3)
predict(mod, a, predleaf=T)
predict(mod, a, reshape=TRUE)
