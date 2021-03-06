% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/deepforest.R
\name{predict.deepforest}
\alias{predict.deepforest}
\title{Deep Forest Prediction}
\usage{
\method{predict}{deepforest}(object, newdata, nlayer = object$nlayer,
  accumulate = object$accumulate, reshape = TRUE)
}
\arguments{
\item{object}{A \code{deepforest} object}

\item{newdata}{An object to predict on}

\item{nlayer}{The number of layers to use for prediction}

\item{accumulate}{A logical indicating whether to use the accumulated predictions.}

\item{reshape}{If \code{reshape} then the means of each observation across meta models are
computed and returned.}
}
\value{
A matrix of the predictions
}
\description{
Predict function for a \code{\link{deepforest}} object.
}
