#'@title Deep Forest Summary
#'
#'@description A summary of the deepforest model
#'
#'@export
summary.deepforest <- function(object) {
    stopifnot("deepforest" %in% class(object))

    cat("Deep Forest Model:\n")
    cat(paste0("  Number of Layers: ", object$nlayer, "\n"))
    cat(paste0("  Number of Meta (repetitions): ", object$nmeta, " (", object$nmetarep, ")\n"))
    cat("  Eval Results:\n------------------------\n")
    dimnames(object$eval) <- list(paste0("Layer", 1:dim(object$eval)[1]),
                                  paste0("Meta", 1:dim(object$eval)[2]),
                                  paste0("Fold", 1:dim(object$eval)[3]))

    print(object$eval)
}

#'@title Deep Forest Plot
#'
#'@description A plot of the deepforest model performance
#'
#'@export
plot.deepforest <- function(object) {
    plot(apply(mod$eval,1,mean))
}

