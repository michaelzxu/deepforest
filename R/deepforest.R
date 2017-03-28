deepforest <- function(x,
                       y,
                       x_test = NULL,
                       y_test = NULL,
                       loss = "MSE",
                       metric = Metrics::mse,
                       nlayer = 5,
                       #Forest
                       nforest = 4,
                       ntree = 100,
                       rnd = 10,
                       nclass = 2,
                       #Tree
                       tree = "random",
                       treeparam = list(),
                       recycle = TRUE,
                       #Weights
                       init = 0.5,
                       batch_size = 1,
                       optimizer = "GradientDescent",
                       optparam = list(learning_rate = 0.1)) {
    .D <- as.matrix(x)
    if (is.null(x_test)){
        .T <- .D
        y_test <- y
    } else {
        .T <- as.matrix(x_test)
    }
    if (is.null(colnames(.D))) {
        colnames(.D) <- paste0("X", 1:dim(.D)[2])
    }
    if (is.null(colnames(.T))) {
        colnames(.T) <- paste0("X", 1:dim(.T)[2])
    }
    .W <- array(init, dim = c(nlayer, nforest, ntree * nclass))
    .M <- vector("list", nlayer)
    for (.l in 1:nlayer) {
        .D.l <- matrix(0, nrow = dim(x)[1], ncol = as.integer(nforest * nclass))
        .T.l <- matrix(0, nrow = dim(x_test)[1], ncol = as.integer(nforest * nclass))
        .Forest <- rep(list(vector("list", ntree)), nforest)
        for (.f in 1:nforest) {
            cat(paste0("Layer: ", .l, ", Forest: ", .f, ", Progress: |-"))
            .prog <- 0
            # if (.l <= 2 | !recycle) {
            .tree <- .tree_param(.D, ntree, c(0.5,1))
            # }
            for (.t in 1:ntree) {
                .prog <- .prog + 1
                .Forest[[.f]][[.t]] <- .grow_tree(.D[.tree[[1]][ ,.t],
                                                     .tree[[2]][ ,.t]],
                                                  y[.tree[[1]][ ,.t]],rnd)
                if (.prog %% ceiling(ntree/10) == 0) cat("-")
            }
            cat("| done!\n")
            cat("  Optimizing weights, Progress: |-")
            .prog <- 0
            # for (.i in sample(dim(x)[1])) {
            #     .prog <- .prog + 1
            #     if (.prog %% floor(dim(.D)[1] / 10) == 0) cat("-")
            #     .W[.l, , ] <- .optimize(.D[.i, ], y[.i], .W[.l, , ], .Forest)
            # }
            .pred <- .predict(.Forest[[.f]], .D)
            .pred.t <- .predict(.Forest[[.f]], .T)
            # .W[.l, .f, ] <- as.numeric(optimx::optimx(.W[.l,.f, ],method = "BFGS",
            #                                fn = function(x) Metrics::mse(y,.predict(.Forest[[.f]], .D, x)))[,1:ntree])
            # .W[.l, .f, ] <- sgd::sgd(.pred,y=y,model="lm")[["coefficients"]]
            .W[.l, .f, ] <- rep(1, ntree * nclass)
            cat("| done!\n")
            #.pred <- .predict(.Forest[[.f]], .D)
            #cat(paste0("  Loss: ", .printtab(.optloss(.pred, y, loss))))
            #cat(paste0("  Eval: ", .printtab(.metric(.pred, y, metric))))
            cat(paste0("  Eval: ",metric(y_test, .pred.t %*% .W[.l, .f, ])))
            .D.l[ ,((.f - 1) * nclass + 1):(.f * nclass)] <- .pred %*% .W[.l,.f, ]
            .T.l[ ,((.f - 1) * nclass + 1):(.f * nclass)] <- .pred.t %*% .W[.l,.f, ]
            .M[[.l]] <- .Forest
            cat("\n")
        }
        .D <- .D.l
        .T <- .T.l
        colnames(.D) <- paste0("X", 1:dim(.D)[2])
        colnames(.T) <- paste0("X", 1:dim(.T)[2])
    }
    return(structure(list(forests = .M, weights = .W), class = "deepforest"))
}

.tree_param <- function(data, n, sample_fraction = rep(0.8, length(dim(data))),
                       balance = FALSE) {
    out <- vector("list", length(dim(data)))
    for (i in 1:length(dim(data))) {
        out[[i]] <- sapply(1:n, function(x) {
            sample(dim(data)[i])[1:floor(sample_fraction[i] * dim(data)[i])]
        })
    }
    return(out)
}

.grow_tree <- function(x, y, rnd) {
    x_dm <- xgboost::xgb.DMatrix(data=x, label=y)
    return(xgboost::xgb.train(data=x_dm, nrounds=1, verbose=0, objective="binary:logistic",
                              params=list(booster="gbtree",eta=0.1,max_depth=99999,
                                          max_leaves=99999,colsample_bytree=1,colsample_bylevel=0.5,
                                          num_parallel_tree=rnd)))
    # return(C50::C5.0(x = x, y = y))
}

.predict <- function(object, newdata, weights = rep(1, length(object))) {
    # lev <- length(object[[1]][["levels"]])
    out <- matrix(nrow = nrow(newdata),
                  ncol = length(object))
    # cat("  ")
    for (i in 1:length(object)) {
        # cat(paste0(i,".."))
        out[ ,i] <- predict(object[[i]], newdata)
    }
    if (all(unique(weights) != 1)) {
        out <- (out %*% weights) / sum(weights)
    }
    return(out)
}

.optimize <- function(x, y, weights, forest) {
    sgd::sgd()
}

predict.deepforest <- function(dfobject, newdata) {
    .D <- as.matrix(newdata)
    for (.l in 1:length(dfobject$forests)) {
        .D.l <- matrix(nrow = nrow(newdata), ncol = length(dfobject$forests[[.l]]))
        for (.f in 1:length(dfobject$forests[[.l]])) {
            cat(paste0("Layer: ",.l,", Forest: ",.f,"\n"))
            .pred <- .predict(dfobject$forests[[.l]][[.f]], .D, dfobject$weights[.l, .f, ])
            .D.l[ , .f] <- .pred
        }
        .D <- .D.l
        colnames(.D) <- paste0("X", 1:dim(.D)[2])
    }
    return(rowSums(.D) / ncol(.D))
}

mod<-deepforest(x=as.matrix(agaricus.train$data),y=agaricus.train$label,x_test = as.matrix(agaricus.test$data), y_test = agaricus.test$label,
                nclass=1,nforest=5,ntree=10,rnd=10)
dfout<-predict(mod, as.matrix(agaricus.test$data))
Metrics::auc(agaricus.test$label,dfout)
Metrics::logLoss(agaricus.test$label,dfout)

xgb<-xgboost::xgboost(data=agaricus.train$data,label=agaricus.train$label,nround=100)
dfout2<-predict(xgb, agaricus.test$data)
Metrics::auc(agaricus.test$label,dfout2)
Metrics::logLoss(agaricus.test$label,dfout2)


credc <- read.csv("../../creditcard.csv")
intrain <- sample(nrow(credc))[1:as.integer(nrow(credc)*0.5)]
credc.train <- credc[intrain,which(!names(credc)=="Class")]
credc.test <- credc[-intrain,which(!names(credc)=="Class")]

credcmod<-deepforest(x=as.matrix(credc.train),y=credc[intrain,"Class"],
                     x_test=as.matrix(credc.test),y_test=credc[-intrain,"Class"],
                     nlayer=5,nclass=1,nforest=5,ntree=100,rnd=1,metric=Metrics::auc)
hout<-predict(credcmod, as.matrix(credc.test))
Metrics::auc(credc[-intrain,"Class"],hout)
Metrics::mse(credc[-intrain,"Class"],hout)
caret::confusionMatrix(credc[-intrain,"Class"],as.integer(round(hout)))

xgb<-xgboost::xgboost(data=as.matrix(credc.train),label=credc[intrain,"Class"],nround=20)
hout2<-predict(xgb,as.matrix(credc.test))
Metrics::auc(credc[-intrain,"Class"],hout2)
Metrics::mse(credc[-intrain,"Class"],hout2)
caret::confusionMatrix(credc[-intrain,"Class"],as.integer(round(hout2)))
