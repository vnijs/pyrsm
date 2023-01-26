library(radiant.data)
library(reticulate)

convert <- function(fl, dir) {
    path <- file.path("data", dir)
    if (!dir.exists(path)) dir.create(path, recursive = TRUE)
    for (f in fl) {
        print(paste0("Working on:", f))
        df <- load(f) %>% get()
        pkl <- r_to_py(df)
        fpy <- sub("\\.rda$", ".pkl", basename(f))
        fpy <- file.path(path, fpy)
        descr <- attr(df, "description")
        if (!is.null(descr)) {
            py_set_attr(pkl, "description", descr)
            pkl[["_metadata"]]$append("description")
        }
        py_save_object(pkl, fpy)
    }
}


## run in container - doesn't like setup on macOS for some reason
fl <- list.files("../radiant.data/data", pattern = "\\.rda$", full.names = TRUE, recursive = TRUE)
convert(fl, "data")
fl <- list.files("../radiant.basics/data", pattern = "\\.rda$", full.names = TRUE, recursive = TRUE)
convert(fl, "basics")
fl <- list.files("../radiant.design/data", pattern = "\\.rda$", full.names = TRUE, recursive = TRUE)
convert(fl, "design")
fl <- list.files("../radiant.model/data", pattern = "\\.rda$", full.names = TRUE, recursive = TRUE)
convert(fl, "model")
fl <- list.files("../radiant.multivariate/data", pattern = "\\.rda$", full.names = TRUE, recursive = TRUE)
convert(fl, "multivariate")
