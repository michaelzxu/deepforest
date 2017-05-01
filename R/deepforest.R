#'@title Deep Forest
#'
#'@description
#'A combination of stacking (per Kaggle) and the deep forest idea proposed in
#'https://arxiv.org/pdf/1702.08835.pdf with a few additional knobs and levers
#'of my own..
#'
#'@param x
#'Object coercible to a matrix.
#'
#'@param y
#'A vector of labels for \code{x}.
#'
#'@param x_val
#'Object coercible as matrix used for validation. Defaults to  \code{x} if not
#'specified.
#'
#'@param y_val
#'A vector of labels for \code{x_val}.
#'
#'@param nfold
#'The number of folds to stack on. If \code{index} is specified, this argument
#'is ignored.
#'
#'@param index
#'A list of integers indicating observations belonging to each fold. Random
#'sampling is used if \code{NULL} to create \code{nfold} folds.
#'
#'@param objective
#'The objective passed to \code{\link{xgb.train}}. It is inferred by nclass if
#'not specified.
#'
#'@param nlayer
#'The number of layers to stack.
#'
#'@param nmeta
#'The number of meta models to build on each layer.
#'
#'@param nmetarep
#'The number of repeated meta models to construct.
#'
#'@param nclass
#'An integer indicating the number of classes. \code{1} for regression, \code{2}
#'for binary classification and \code{>2} for multiclass. If \code{NULL}, then
#'it is set to \code{length(unique(y))} if the value is less than 10 and 1
#'otherwise.
#'
#'@param metatype
#'A list of character variables indicating type of meta models to construct.
#'Currently supports "bag", "boost", "dart" and "lin" for a random forest,
#'boosted trees, boosted trees with dropout and boosted linear models
#'respectively, all based on xgboost. If length of \code{metatype} is not equal
#'to \code{nmeta * nmetarep} then it is recycled to that length.
#'
#'@param metaparam
#'A list indicating additional parameters for each meta model. The mechanism
#'used to generate default values and create randomized values are in
#'\code{\link{metaparam}}. If list is shorter than
#'\code{nmeta * nmetarep} then all meta models without a clear \code{metaparam}
#'will take on default values, so you should indicate manually tuned models
#'first in \code{metatype}.
#'
#'@param metarandom
#'A logical indicating whether to create randomized miscellaneous parameters.
#'
#'@param colsample_bylayer
#'A numeric vector between [0,1] indicating the fraction of features to sample
#'at each layer. If a single value, then you may use \code{colsample_add} to
#'further modify how the value changes, otherwise a constant value will be used
#'for each layer.
#'
#'@param colsample_add
#'Either a logical or a numeric. If \code{TRUE} then values will be added at
#'each layer to ensure that the final \code{nlayer} has a sampling rate of 1
#'(i.e. all features used). If \code{FALSE} then no modifier is applied to
#'\code{colsample_bylayer}. If a numeric value, that constant value is added
#'to \code{colsample_bylayer} after each layer.
#'
#'@param accumulate
#'A logical indicating if the predictions by meta models from each layer is
#'accumulated. If \code{FALSE} then each layer will not have the direct
#'predictions from two layers before although it will be implicitly
#'approximated by the predictions from the layer immediately before.
#'
#'@param nthread
#'The nthread argument passed to \code{\link{xgb.train}}.
#'
#'@param eval_func
#'A function used to evaluate each meta model.
#'
#'@param missing
#'The \code{missing} argument passed to \code{\link{xgb.train}}.
#'
#'@param ...
#'Additional arguments passed to \code{\link{xgb.train}}.
#'
#'@return
#'A \code{deepforest} object containing all the models constructed and sampling
#'results for used in prediction on test data.
#'
#'@export
deepforest <- function(x, y, x_val = NULL, y_val =NULL, nfold = 5, index = NULL,
                       objective = NULL, eval_func = NULL,
                       nlayer = 5, nmeta = 4, nmetarep = 1, nclass = NULL,
                       metatype = rep(c("bag", "boost", "dart", "lin"),
                                      length = nmeta * nmetarep),
                       metaparam = NULL, metarandom = FALSE,
                       colsample_bylayer = 1, colsample_add = FALSE,
                       accumulate = FALSE, nthread = 2, missing = NA,
                       printby = "fold", ...) {

    stopifnot(length(metatype) %in% c(nmeta, nmeta * nmetarep))
    stopifnot(any(length(colsample_bylayer == 1),
                  length(colsample_bylayer) == nlayer))
    stopifnot(is.numeric(y))
    stopifnot(is.null(y_val) | is.numeric(y_val))

    #Recycle metatype
    if (length(metatype) != nmeta * nmetarep) {
        metatype = rep(metatype, length = nmeta * nmetarep)
    }

    #Create folds
    if (is.null(index)) {
        index <- deepforest::getFolds(y, nfold = nfold, list = TRUE)
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
                colsample_bylayer <- rep(colsample_bylayer, nlayer)
            }
        }
    }
    colsample_bylayer <- pmin(1, colsample_bylayer)

    #Determine task
    if (is.null(nclass)) {
        if (length(unique(y)) < 10) {
            nclass <- length(unique(y))
        } else {
            nclass <- 1
        }
    }

    #Default metrics - taken from Metrics package
    if (is.null(eval_func)) {
        if (nclass == 1) {
            eval_metric <- "rmse"
            eval_func <- function(actual, predicted) {
                sqrt(mean((actual - predicted)^2))
            }
        } else if (nclass == 2) {
            eval_metric <- "auc"
            eval_func <- function(actual, predicted) {
                r <- rank(predicted)
                n_pos <- sum(actual==1)
                n_neg <- length(actual) - n_pos
                auc <- (sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
                auc
            }
        } else if (nclass > 2) {
            eval_metric <- "logloss"
            eval_func <- function (actual, predicted) {
                score <- -(actual * log(predicted) +
                               (1 - actual) * log(1 - predicted))
                score[actual == predicted] <- 0
                score[is.nan(score)] <- Inf
                score
            }
        }
    } else {
        eval_metric <- as.character(quote(eval_func))
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
        metaparam <- deepforest::metaparam(metatype, metarandom = metarandom)
    } else {
        if (length(metaparam) != length(metatype)) {
            metaparam <- c(metaparam, rep(list(list()), length(metatype) -
                                              length(metaparam)))
        }
        metaparam <- deepforest::metaparam(metatype, metaparam,
                                           metarandom = metarandom)
    }

    #Copy original x
    orig_x <- x
    orig_x_val <- x_val

    .model <- rep(list(rep(list(vector("list", nfold)), nmeta * nmetarep)),
                  nlayer)
    .eval <- array(dim = c(nlayer, nmeta * nmetarep, nfold))
    .wgts <- list()
    .wgts.acc <- list()
    .colsamp <- vector("list", nlayer)

    for (.l in 1:nlayer) {
        .lcols <- sample(ncol(orig_x))[
            1:max(1, floor(ncol(orig_x) * colsample_bylayer[.l]))]
        .colsamp[[.l]] <- .lcols

        if (.l == 1 | !accumulate) {
            x <- orig_x[ , .lcols]
            x_val <- orig_x_val[ , .lcols]
        } else {
            if (accumulate) {
                x <- cbind(.x.l, orig_x[ , .lcols])
                x_val <- cbind(.x_val.l, orig_x_val[ , .lcols])
            } else {
                x <- cbind(.x.tmp, orig_x[ ,.lcols])
                x_val <- cbind(.x_val.tmp, orig_x_val[ ,.lcols])
            }
        }

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
                metaparam[[.m]][["colsample_bytree"]] <-
                    max(metaparam[[.m]][["colsample_bytree"]], 2 / ncol(x))
                if (nclass <= 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = metaparam[[.m]][["nrounds"]],
                        objective =  ifelse(!is.null(objective),objective,ifelse(nclass == 1, "reg:linear",
                                            "binary:logistic")),
                        verbose = FALSE, nthread = nthread,
                        params = metaparam[[.m]], ...)
                    .x.l[index[[.f]], .m] <- predict(.model[[.l]][[.m]][[.f]],
                                                     x[index[[.f]], ])
                } else if (nclass > 2) {
                    .model[[.l]][[.m]][[.f]] <- xgboost::xgb.train(
                        data = x.f, nrounds = metaparam[[.m]][["nrounds"]],
                        objective = ifelse(!is.null(objective),objective,"multi:softprob"), verbose = FALSE,
                        nthread = nthread, params = metaparam[[.m]], ...)
                    .x.l[index[[.f]], ((.m - 1) * nclass + 1):(.m * nclass)] <-
                        predict(.model[[.l]][[.m]][[.f]], x[index[[.f]], ],
                                reshape = TRUE)
                } else {
                    stop("Invalid nclass")
                }
                if (nclass > 2) {
                    .eval[.l, .m, .f] <- eval_func(y_val,apply(
                        predict(.model[[.l]][[.m]][[.f]], x_val.m,
                                reshape = TRUE),
                        1, which.max))
                } else {
                    .eval[.l, .m, .f] <-
                        eval_func(y_val, predict(.model[[.l]][[.m]][[.f]],
                                                 x_val.m))
                }
                if (printby == "fold") {
                    cat(paste0(", ", eval_metric, " - ", .eval[.l, .m, .f],
                               "\n"))
                }
                gc()
            }
            if (printby == "meta") {
                cat(paste0(", ", eval_metric, " - ", mean(.eval[.l, .m, ]),
                           "\n"))
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
            .x.tmp <- .x.l
            .x_val.tmp <- .x_val.l
        } else {
            .x.tmp <- cbind(.x.tmp, .x.l)
            .x_val.tmp <- cbind(.x_val.tmp, .x_val.l)
        }

        if (nclass == 1) {
            .wgts <- c(.wgts, list(stats::glm.fit(x = .x.l, y = y, intercept = FALSE)$coefficients))
            .wgts.acc <- c(.wgts.acc, list(stats::glm.fit(x = .x.tmp, y = y, intercept = FALSE)$coefficients))
        } else {
            .wgts <- c(.wgts, list(stats::glm.fit(x = .x.l, y = y, intercept = FALSE, family = binomial())$coefficients))
            .wgts.acc <- c(.wgts.acc, list(stats::glm.fit(x = .x.tmp, y = y, intercept = FALSE, family = binomial())$coefficients))
        }

        cat("------------------------------------------------\n")
        cat(paste0("Layer ", .l, ", Average ", eval_metric, " - ",
                   eval_func(y_val, .x_val.l %*% .wgts[[length(.wgts)]]), "\n"))
        cat(paste0("Layer ", .l, ", Accumed ", eval_metric, " - ",
                   eval_func(y_val, .x_val.tmp %*% .wgts.acc[[length(.wgts.acc)]]), "\n"))
        cat("------------------------------------------------\n")
        gc()
    }
    return(structure(list(model = .model, colsamp = .colsamp, eval = .eval,
                          nlayer = nlayer, nmeta = nmeta, nmetarep = nmetarep,
                          nclass = nclass, accumulate = accumulate,
                          weights = list(noacc = .wgts, acc = .wgts.acc),
                          metaparam = metaparam, metatype = metatype),
                     class = "deepforest"))
}

#'@title Deep Forest Prediction
#'
#'@description
#'Predict function for a \code{\link{deepforest}} object.
#'
#'@param object
#'A \code{deepforest} object
#'
#'@param newdata
#'An object to predict on
#'
#'@param nlayer
#'The number of layers to use for prediction
#'
#'@param accumulate
#'A logical indicating whether to use the accumulated predictions.
#'
#'@param reshape
#'If \code{reshape} then the means of each observation across meta models are
#'computed and returned.
#'
#'@return
#'A matrix of the predictions
#'
#'@export
predict.deepforest <- function(object, newdata, nlayer = object$nlayer,
                               accumulate = object$accumulate, reshape = TRUE) {

    stopifnot(all(c("model", "colsamp", "eval", "nlayer", "nmeta", "nmetarep",
                    "nclass", "accumulate", "metaparam") %in% names(object)))

    newdata <- as.matrix(newdata)
    if (is.null(colnames(newdata))) {
        colnames(newdata) <- paste0("X", 1:ncol(newdata))
    }

    for (.l in 1:nlayer) {
        if (.l == 1 | !accumulate) {
            x <- newdata[ , object$colsamp[[.l]]]
        } else {
            if (accumulate) {
                x <- cbind(.x.l, newdata[ , object$colsamp[[.l]]])
            } else {
                x <- cbind(.x.tmp, newdata[ , object$colsamp[[.l]]])
            }
        }

        .colnm <- paste0("L", .l, "_M", rep(1:(object$nmeta * object$nmetarep),
                                            each = ifelse(object$nclass == 2, 1,
                                                          object$nclass)),
                         "_C", rep(1:ifelse(object$nclass == 2, 1,
                                            object$nclass),
                                   object$nmeta * object$nmetarep))
        .x.l <- matrix(nrow = nrow(newdata), ncol = object$nmeta *
                           object$nmetarep * ifelse(object$nclass == 2, 1,
                                                    object$nclass),
                       dimnames = list(NULL, .colnm))
        for (.m in 1:(object$nmeta * object$nmetarep)) {
            x.m <- xgboost::xgb.DMatrix(data = x, missing = missing)
            if (object$nclass <= 2) {
                .x.l[ , .m] <-
                    apply(simplify2array(lapply(object$model[[.l]][[.m]],
                                                predict, newdata = x.m)),
                          1, mean)
            } else if (object$nclass > 2) {
                .x.l[ , ((.m - 1) * object$nclass + 1):(.m * object$nclass)] <-
                    apply(simplify2array(lapply(object$model[[.l]][[.m]],
                                                predict, newdata = x.m)),
                          1:object$nmeta, mean)
            }
        }

        if (.l == 1) {
            .x.tmp <- .x.l
        } else {
            .x.tmp <- cbind(.x.tmp, .x.l)
        }
    }

    if (reshape) {
        if (accumulate) {
            return(.x.tmp %*% object$weights$acc[[nlayer]])
        } else {
            return(.x.l %*% object$weights$noacc[[nlayer]])
        }
    } else {
        if (accumulate) {
            return(.x.tmp)
        } else {
            return(.x.l)
        }
    }
}


