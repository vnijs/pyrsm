from scipy.stats import norm
import math


class sample_size:
    """
    Sample size calculation

    Parameters
    ----------
    type: Choose "mean" or "proportion"
    err_mean Acceptable Error for Mean
    sd_mean Standard deviation for Mean
    err_prop Acceptable Error for Proportion
    p_prop Initial proportion estimate for Proportion
    conf Confidence level
    pop_size Population size
    incidence Incidence rate (i.e., fraction of valid respondents)
    response Response rate

    @return A list of variables defined in sample_size as an object of class sample_size

    sample_size(type = "mean", err_mean = 2, sd_mean = 10)
    """

    def sample_size_mean(
        type,
        err_mean=2,
        sd_mean=10,
        err_prop=0.1,
        p_prop=0.5,
        conf=0.95,
        pop_size=None,
        incidence=1,
        response=1,
    ):
        if conf is None or conf < 0 or conf > 1:
            conf = 0.95

        zval = -norm.ppf((1 - conf) / 2)

        if type == "mean":
            if err_mean is None:
                return "Please select an acceptable error greater than 0"
            n = (zval**2 * sd_mean**2) / (err_mean**2)
        else:
            if err_prop is None:
                return "Please select an acceptable error greater than 0"
            n = (zval**2 * p_prop * (1 - p_prop)) / (err_prop**2)

        if pop_size is not None:
            n = n * pop_size / ((n - 1) + pop_size)

        return math.ceil(n)


# Test the function
sample_size("mean", err_mean=2, sd_mean=10)

# summary.sample_size <- function(object, ...) {
#   if (is.character(object)) {
#     return(object)
#   }

#   cat("Sample size calculation\n")

#   if (object$type == "mean") {
#     cat("Calculation type     : Mean\n")
#     cat("Acceptable Error     :", object$err_mean, "\n")
#     cat("Standard deviation   :", object$sd_mean, "\n")
#   } else {
#     cat("Calculation type     : Proportion\n")
#     cat("Acceptable Error     :", object$err_prop, "\n")
#     cat("Proportion           :", object$p_prop, "\n")
#   }

#   cat("Confidence level     :", object$conf_lev, "\n")
#   cat("Incidence rate       :", object$incidence, "\n")
#   cat("Response rate        :", object$response, "\n")

#   if (object$pop_correction == "no") {
#     cat("Population correction: None\n")
#   } else {
#     cat("Population correction: Yes\n")
#     cat("Population size      :", format_nr(object$pop_size, dec = 0), "\n")
#   }

#   cat("\nRequired sample size     :", format_nr(object$n, dec = 0))
#   cat("\nRequired contact attempts:", format_nr(ceiling(object$n / object$incidence / object$response), dec = 0))

#   rm(object)
# }
