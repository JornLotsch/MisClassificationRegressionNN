#################################### Libraries ########################################################################

library(parallel)
library(cowplot)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(ggforce)
library(randomForest)
library(MASS)
library(RColorBrewer)
library(caret)
library(ComplexHeatmap)
library(e1071)
library(nnet)
library(doParallel)


#################################### Functions ########################################################################
plot_rawData <- function(Data) {
    colors_a <- colors[as.numeric(Data$Cls)]
    if (ncol(within(Data, rm(Cls))) == 3) {
        scatterplot3d(within(Data, rm(Cls))[, 1], within(Data, rm(Cls))[, 2], within(Data, rm(Cls))[, 3],
            color = colors_a, pch = 16,
            xlab = "X", ylab = "Y", zlab = "Z"
        )
    } else {
        if (ncol(within(Data, rm(Cls))) == 2) {
            plot(within(Data, rm(Cls))[, 1] ~ within(Data, rm(Cls))[, 2], col = colors_a, pch = 16, xlab = "X", ylab = "Y")
        } else {
            plot(within(Data, rm(Cls))[, 1], col = colors_1t, pch = 16, xlab = "X")
        }
    }
    pic <- grab_grob()
    return(pic)
}

grab_grob <- function() {
    grid.echo()
    grid.grab()
}


multiClassSummary <- function(data, lev = NULL, model = NULL) {
    # Load Libraries
    require(Metrics)
    require(caret)

    # Check data
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) {
        stop("levels of observed and predicted data do not match")
    }

    # Calculate custom one-vs-all stats for each class
    prob_stats <- lapply(levels(data[, "pred"]), function(class) {
        # Grab one-vs-all data for the class
        pred <- ifelse(data[, "pred"] == class, 1, 0)
        obs <- ifelse(data[, "obs"] == class, 1, 0)
        prob <- data[, class]

        # Calculate one-vs-all AUC and logLoss and return
        cap_prob <- pmin(pmax(prob, .000001), .999999)
        prob_stats <- c(auc(obs, prob), logLoss(obs, cap_prob))
        names(prob_stats) <- c("ROC", "logLoss")
        return(prob_stats)
    })
    prob_stats <- do.call(rbind, prob_stats)
    rownames(prob_stats) <- paste("Class:", levels(data[, "pred"]))

    # Calculate confusion matrix-based statistics
    CM <- confusionMatrix(data[, "pred"], data[, "obs"])

    # Aggregate and average class-wise stats
    # Todo: add weights
    class_stats <- cbind(CM$byClass, prob_stats)
    class_stats <- colMeans(class_stats)

    # Aggregate overall stats
    overall_stats <- c(CM$overall)

    # Combine overall with class-wise stats and remove some stats we don't want
    stats <- c(overall_stats, class_stats)
    stats <- stats[!names(stats) %in% c(
        "AccuracyNull",
        "Prevalence", "Detection Prevalence"
    )]

    # Clean names and return
    names(stats) <- gsub("[[:blank:]]+", "_", names(stats))
    return(stats)
}


TrainRF <- function(TrainingTestDataCls, ValidationDataCls, method = "RF") {
    TrainingTestDataCls$Cls <- factor(make.names(TrainingTestDataCls$Cls))
    TrainingTestDataCls$Cls <- droplevels(TrainingTestDataCls$Cls)

    model_weights_global <- 1 / table(TrainingTestDataCls$Cls)
    model_weights <- model_weights_global[TrainingTestDataCls$Cls]

    fitControl <- trainControl(
        method = "repeatedcv",
        number = 20,
        repeats = 5,
        ## Estimate class probabilities
        classProbs = TRUE,
        summaryFunction = multiClassSummary,
        allowParallel = TRUE
    )
    cl <- makePSOCKcluster(nProc)
    registerDoParallel(cl)

    switch(method,
        SVM = {
            TrainedClassifier_svmLinear <- train(Cls ~ .,
                data = TrainingTestDataCls,
                method = "svmLinear",
                importance = T,
                trControl =
                    fitControl,
                verbose = FALSE,
                weights = model_weights,
                metric = "Balanced_Accuracy",
                preProcess = c("center", "scale", "YeoJohnson"),
                tuneLength = 5
            )
            TrainedClassifier_svmRadial <- train(Cls ~ .,
                data = TrainingTestDataCls,
                method = "svmRadial",
                importance = T,
                trControl =
                    fitControl,
                verbose = FALSE,
                weights = model_weights,
                metric = "Balanced_Accuracy",
                preProcess = c("center", "scale", "YeoJohnson"),
                tuneLength = 5
            )
            TrainedClassifier_svmPoly <- train(Cls ~ .,
                data = TrainingTestDataCls,
                method = "svmPoly",
                importance = T,
                trControl =
                    fitControl,
                verbose = FALSE,
                weights = model_weights,
                metric = "Balanced_Accuracy",
                preProcess = c("center", "scale", "YeoJohnson"),
                tuneLength = 5
            )
            SVM_variants <- c("TrainedClassifier_svmLinear", "TrainedClassifier_svmRadial", "TrainedClassifier_svmPoly")

            if ("Balanced_Accuracy" %in% colnames(TrainedClassifier_svmLinear$results)) {
                TrainedClassifier <- get(paste(SVM_variants[which.max(c(
                    mean(TrainedClassifier_svmLinear$results$Balanced_Accuracy),
                    mean(TrainedClassifier_svmRadial$results$Balanced_Accuracy),
                    mean(TrainedClassifier_svmPoly$results$Balanced_Accuracy)
                ))]))
            } else {
                TrainedClassifier <- get(paste(SVM_variants[which.max(c(
                    mean(TrainedClassifier_svmLinear$results$ROC),
                    mean(TrainedClassifier_svmRadial$results$ROC),
                    mean(TrainedClassifier_svmPoly$results$ROC)
                ))]))
            }
        },
        Multinom = {
            if (length(unique(TrainingTestDataCls$Cls)) == 2) {
                TrainedClassifier <-
                    train(Cls ~ .,
                        data = TrainingTestDataCls, "glm",
                        family = "binomial",
                        weights = model_weights,
                        metric = "Balanced_Accuracy",
                        preProcess = c("center", "scale", "YeoJohnson"),
                        tuneLength = 10
                    )
            } else {
                TrainedClassifier <-
                    train(Cls ~ .,
                        data = TrainingTestDataCls, "multinom",
                        weights = model_weights,
                        metric = "Balanced_Accuracy",
                        preProcess = c("center", "scale", "YeoJohnson"),
                        tuneLength = 10
                    )
            }
        },
        TrainedClassifier <- train(Cls ~ .,
            data = TrainingTestDataCls,
            method = method,
            trControl =
                fitControl,
            weights = model_weights,
            metric = "Balanced_Accuracy",
            preProcess = c("center", "scale", "YeoJohnson"),
            tuneLength = 10
        )
    )

    stopCluster(cl)
    registerDoSEQ()

    pred <- predict(TrainedClassifier, newdata = ValidationDataCls)
    pred2 <- predict(TrainedClassifier, newdata = ValidationDataCls, type = "prob")

    return(list(pred = pred, pred2 = pred2, TrainedClassifier = TrainedClassifier))
}
