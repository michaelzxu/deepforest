deepforest <- function(x,
                       y,
                       loss = "MSE",
                       metric = "error",
                       nlayer = 5,
                       #Forest
                       nforest = 4,
                       ntree = 100,
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
    if (is.null(colnames(.D))) {
        colnames(.D) <- paste0("X", 1:dim(.D)[2])
    }
    .W <- array(init, dim = c(nlayer, nforest, ntree * nclass))
    .M <- vector("list", nlayer)
    for (.l in 1:nlayer) {
        .D.l <- matrix(0, nrow = dim(x)[1], ncol = as.integer(nforest * nclass))
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
                                                  as.factor(y[.tree[[1]][ ,.t]]))
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
            # .W[.l, .f, ] <- as.numeric(optimx::optimx(.W[.l,.f, ],method = "BFGS",
            #                                fn = function(x) Metrics::mse(y,.predict(.Forest[[.f]], .D, x)))[,1:ntree])
            .W[.l, .f, ] <- sgd::sgd(.pred,y=y,model="lm")[["coefficients"]]
            cat("| done!\n")
            #.pred <- .predict(.Forest[[.f]], .D)
            #cat(paste0("  Loss: ", .printtab(.optloss(.pred, y, loss))))
            #cat(paste0("  Eval: ", .printtab(.metric(.pred, y, metric))))
            cat(paste0("  Eval: ",Metrics::mse(y, .pred %*% .W[.l, .f, ])))
            .D.l[ ,((.f - 1) * nclass + 1):(.f * nclass)] <- .pred %*% .W[.l,.f, ]
            .M[[.l]] <- .Forest
            cat("\n")
        }
        .D <- .D.l
        colnames(.D) <- paste0("X", 1:dim(.D)[2])
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

.grow_tree <- function(x, y) {
    return(C50::C5.0(x = x, y = y))
}

.predict <- function(object, newdata, weights = rep(1, length(object))) {
    # lev <- length(object[[1]][["levels"]])
    out <- matrix(nrow = nrow(newdata),
                  ncol = length(object))
    cat("  ")
    for (i in 1:length(object)) {
        cat(paste0(i,".."))
        out[ ,i] <- predict(object[[i]], newdata,type="prob")[,2]
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
    .D <- newdata
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
    return(rowSums(.D)/ncol(.D))
}

mod<-deepforest(x=as.matrix(agaricus.train$data),y=agaricus.train$label,nclass=1,nforest=5,ntree=100)
predict(mod, as.matrix(agaricus.test$data))
dfout<-predict(mod, as.matrix(agaricus.test$data))

iris_train<-sample(150)[1:100]
mod2<-deepforest(x=as.matrix(iris[iris_train,1:4]),y=as.integer(iris[iris_train,5]=="setosa"),nclass=1,nforest=5,ntree=10,nlayer=3)
dfout2<-predict(mod2, iris[-iris_train,1:4])
round(dfout2)
as.integer(iris[-iris_train,5]=="setosa")
