metaparam <- function (metatype, param = rep(list(list()), length(metatype)),
                       metarandom = FALSE) {
    .i <- 0
    for (meta in metatype) {
        .i <- .i + 1
        if (meta == "bag") {
            .chkdef <- list(booster = "gbtree",
                            nrounds = 1,
                            num_parallel_tree = 1000,
                            subsample = 0.632,
                            colsample_bytree = 0.2,
                            max_depth = 99999,
                            max_leaves = 99999,
                            eta = 1)
            .chkrnd <- list(gamma = runif(1, 0, 0.2),
                            min_child_weight = runif(1, 0, 100),
                            max_delta_step = rgamma(1, 1, 0.33),
                            lambda = runif(1, 1, 10),
                            alpha = runif(1, 0, 5),
                            num_parallel_tree = runif(1, 200,1000),
                            subsample = runif(1, 0.4, 1),
                            colsample_bytree = runif(1, 0.1, 1))
        } else if (meta %in% c("dart","boost")) {
            .chkdef <- list(booster = ifelse(meta == "dart", "dart", "gbtree"),
                            nrounds = 100,
                            subsample = 0.632,
                            colsample_bytree = 0.6,
                            eta = 0.05,
                            max_depth = 8)
            .chkrnd <- list(gamma = runif(1, 0, 0.2),
                            min_child_weight = runif(1, 0, 100),
                            max_delta_step = rgamma(1, 1, 0.33),
                            lambda = runif(1, 1, 10),
                            alpha = runif(1, 0, 5),
                            num_parallel_tree = runif(1, 200,1000),
                            subsample = runif(1, 0.4, 1),
                            colsample_bytree = runif(1, 0.1, 1),
                            nrounds = runif(1, 10, 500),
                            eta = runif(1, 0.001, 0.3))
            if (meta == "dart") {
                .chkrnd <- c(.chkrnd, rate_drop = runif(1, 0.00001, 0.4),
                             skip_drop = runif(1, 0, 0.05))
            }
        } else if (meta == "lin") {
            .chkdef <- list(booster = "gblinear",
                            nrounds = 100)
            .chkrnd <- list(lambda = rgamma(1, 1, 0.5)^2,
                            alpha = rgamma(1, 1, 0.5)^2)
        }
        for (.chk in names(.chkdef)) {
            if (is.null(param[[.i]][[.chk]])) {
                param[[.i]] <- c(param[[.i]], .chkdef[.chk])
                names(param[[.i]])[length(param[[.i]])] <- .chk
            }
        }
        for (.chk in names(.chkrnd)) {
            if (!is.null(param[[.i]][[.chk]])) {
                if (param[[.i]][[.chk]] == "random") {
                    param[[.i]][[.chk]] <- .chkrnd[.chk]
                } else if (metarandom) {
                    param[[.i]] <- c(param[[.i]], .chkrnd[.chk])
                    names(param[[.i]])[length(param[[.i]])] <- .chk
                }
            }
        }
    }
    return(param)
}

deepforest <- function(x, y, x_val = NULL, y_val =NULL, nfold = 5, index = NULL,
                       nlayer = 5, nmeta = 4, nmetarep = 1, nclass = NULL,
                       metatype = rep(c("bag", "boost", "dart", "lin"),
                                      length = nmeta * nmetarep),
                       metaparam = NULL, metarandom = FALSE,
                       colsample_bylayer = 1, colsample_add = FALSE,
                       accumulate = FALSE, nthread = 2, eval_metric = NULL,
                       missing = NA, printby = "fold",
                       ...) {

    stopifnot(length(metatype) == nmeta * nmetarep)
    stopifnot(any(length(colsample_bylayer == 1),
                  length(colsample_bylayer) == nlayer))

    #Create folds
    if (is.null(index)) {
        index <- caret::createFolds(y, k = nfold, list = TRUE)
    } else {
        if (class(index) != "list") stop("index must be a list")
        nfold <- length(index)
    }

    #Set layer colsample
    if (nlayer > 1 & length(colsample_bylayer) == 1) {
        if (is.numeric(colsample_add)) {
            colsample_bylayer <- seq(from = colsample_bylayer,
                                     by = colsample_add, length.out = nlayer)
        } else if (is.logical(colsample_add)) {
            if (colsample_add) {
                colsample_bylayer <- seq(from = colsample_bylayer,
                                         to = 1, length.out = nlayer)
            } else {
                rep(colsample_bylayer, nlayer)
            }
        }
    }

    #Determine task
    if (is.null(nclass)) {
        if (length(unique(y)) < 10) {
            nclass <- length(unique(y))
        } else {
            nclass <- 1
        }
    }

    #Default metrics
    if (is.null(eval_metric)) {
        if (nclass == 1) {
            eval_metric <- Metrics::mse
        } else if (nclass == 2) {
            eval_metric <- Metrics::auc
        } else if (nclass > 2) {
            eval_metric <- Metrics::logLoss
        }
    }

    #Coerce x
    x <- as.matrix(x)
    if (is.null(colnames(x))) {
        colnames(x) <- paste0("X", 1:ncol(x))
    }

    #Coerce and format x_val
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

    #Get model params
    if (is.null(metaparam)) {
        metaparam <- metaparam(metatype, metarandom = metarandom)
    } else {
        metaparam <- metaparam(metatype, metaparam, metarandom = metarandom)
    }

    #Copy original x
    orig_x <- x
    orig_x_val <- x_val

    .model <- rep(list(rep(list(vector("list", nfold)), nmeta * nmetarep)),
                  nlayer)
    .eval <- array(dim = c(nlayer, nmeta * nmetarep, nfold))
    .colsamp <- vector("list", nlayer)

    # .lcols <- sample(ncol(x))[1:floor(ncol(x) * colsample_bylayer[1])]
    # x <- x[ , .lcols]
    # x_val <- x_val[ , .lcols]

    for (.l in 1:nlayer) {
        .lcols <- sample(ncol(orig_x))[1:floor(ncol(orig_x) * colsample_bylayer[.l])]
        .colsamp[[.l]] <- .lcols

        if (.l == 1 | !accumulate) {
            x <- orig_x[ , .lcols]
            x_val <- orig_x_val[ , .lcols]
        } else {
            x <- cbind(.x.l, orig_x[ , .lcols])
            x_val <- cbind(.x_val.l, orig_x_val[ , .lcols])
        }
        # x <- orig_x[ , .lcols]
        # x_val <- orig_x_val[ , .lcols]

        # .x.l <- matrix(nrow = nrow(x), ncol = 0)
        # .x_val.l <- matrix(nrow = nrow(x_val), ncol = 0)

        .colnm <- paste0("L", .l, "_M", rep(1:(nmeta * nmetarep), each =
                                                ifelse(nclass == 2, 1, nclass)),
                         "_C", rep(1:ifelse(nclass == 2, 1, nclass),
                                   nmeta * nmetarep))
        .x.l <- matrix(nrow = nrow(x), ncol = nmeta * nmetarep * ifelse(
            nclass == 2, 1, nclass), dimnames = list(NULL, .colnm))
        .x_val.l <- matrix(nrow = nrow(x_val), ncol = nmeta * nmetarep * ifelse(
            nclass == 2, 1, nclass), dimnames = list(NULL, .colnm))

        for (.m in 1:(nmeta * nmetarep)) {
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
                if (nclass <= 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = metaparam[[.m]][["nrounds"]],
                        objective =  ifelse(nclass == 1, "reg:linear",
                                            "binary:logistic"),
                        verbose = FALSE, nthread = nthread,
                        params = metaparam[[.m]], ...)
                    .x.l[index[[.f]], .m] <- predict(.model[[.l]][[.m]][[.f]],
                                                     x[index[[.f]], ])
                } else if (nclass > 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = metaparam[[.m]][["nrounds"]],
                        objective = "multi:softprob", verbose = FALSE,
                        nthread = nthread, params = metaparam[[.m]], ...)
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

        # if (.l == 1) {
        #     x.tmp <- .x.l
        #     x_val.tmp <- .x_val.l
        # } else {
        #     x.tmp <- cbind(x.tmp, .x.l)
        #     x_val.tmp <- cbind(x_val.tmp, .x_val.l)
        # }
        # x <- cbind(x.tmp, orig_x[ , .lcols])
        # x_val <- cbind(x_val.tmp, orig_x_val[ , .lcols])

        cat(paste0("Layer ", .l, " average: ",
                   eval_metric(y_val, rowMeans(.x_val.l)), "\n"))
        gc()
    }
    return(structure(list(model = .model, eval = .eval, colsamp = .colsamp),
                     class = "deepforest"))
}

mod <- deepforest(credc.train,credc[intrain,"Class"],
                  credc.test,credc[-intrain,"Class"],
                  nmeta=4, nlayer=10, nfold = 5, printby = "meta",
                  colsample_bylayer = 0.5, accumulate = TRUE,
                  colsample_add = TRUE)


a<-xgboost::xgb.DMatrix(data=as.matrix(iris[,1:4],label=as.numeric(iris[,5]=="setosa")))
mod<-xgboost::xgb.train(data= a, nrounds=1,num_parallel_tree = 3,colsample_bytree=0.25,objective="binary:logistic")
a<-xgboost::xgb.DMatrix(data=as.matrix(iris[,1:4],label=as.integer(iris[,5])))
mod<-xgboost::xgb.train(data= a, nrounds=1,num_parallel_tree = 3,colsample_bytree=0.25,objective="multi:softprob",num_class=3)
predict(mod, a, predleaf=T)
predict(mod, a, reshape=TRUE)
