#'@title Meta Model Parameters
#'
#'@description
#'Used for setting default meta model parameters
#'
#'@export
metaparam <- function (metatype, param = rep(list(list()), length(metatype)),
                       metarandom = FALSE) {
    stopifnot(length(metatype) == length(param))

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
                            max_delta_step = rgamma(1, 0.5, 3),
                            lambda = runif(1, 1, 10),
                            alpha = runif(1, 0, 5),
                            num_parallel_tree = floor(runif(1, 200,1000)),
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
                            max_delta_step = rgamma(1, 0.5, 3),
                            lambda = runif(1, 1, 10),
                            alpha = runif(1, 0, 5),
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
                }
            }  else if (is.null(param[[.i]][[.chk]]) & metarandom) {
                param[[.i]] <- c(param[[.i]], .chkrnd[.chk])
                names(param[[.i]])[length(param[[.i]])] <- .chk
            }
        }
    }
    return(param)
}

#'@title Get n-folds
#'
#'@description
#'Shamelessly taken from \link{\code{caret::createFolds}}
#'https://topepo.github.io/caret/
#'
#'@export
getFolds <- function(y, nfold = 5, recycle = FALSE, list = TRUE) {
    if (is.numeric(y)) {
        cuts <- min(5, max(2, floor(length(y) / nfold)))
        breaks <- unique(quantile(y, probs = seq(0, 1, length = cuts)))
        y <- cut(y, breaks, include.lowest = TRUE)
    }
    if (nfold < length(y)) {
        y <- factor(as.character(y))
        numInClass <- table(y)
        foldVector <- vector(mode = "integer", length(y))
        for (i in 1:length(numInClass)) {
            min_reps <- numInClass[i]%/%nfold
            if (min_reps > 0) {
                spares <- numInClass[i]%%nfold
                seqVector <- rep(1:nfold, min_reps)
                if (spares > 0)
                    seqVector <- c(seqVector, sample(1:nfold, spares))
                foldVector[which(y == names(numInClass)[i])] <- sample(seqVector)
            }
            else {
                foldVector[which(y == names(numInClass)[i])] <- sample(1:nfold,
                                                                       size = numInClass[i])
            }
        }
    }
    else foldVector <- seq(along = y)
    if (list) {
        out <- split(seq(along = y), foldVector)
        names(out) <- paste("Fold", gsub(" ", "0", format(seq(along = out))),
                            sep = "")
    }
    else out <- foldVector
    out
}
