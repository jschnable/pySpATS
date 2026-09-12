# Usage: Rscript benchmarks/compare_r.R input.csv [output_directory] [repeats]
library(SpATS)
stopifnot(as.character(packageVersion("SpATS")) == "1.0.20")
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1])
d$geno <- factor(d$geno)
repeats <- if(length(args)>=3) as.integer(args[3]) else 1L
results <- data.frame()
for(gr in c(FALSE,TRUE)) {
  for(run in seq_len(repeats)) {
    gc()
    elapsed <- system.time(m <- SpATS(response='yield',genotype='geno',genotype.as.random=gr,
        spatial=~PSANOVA(col,row,nseg=c(10,10),nest.div=2),data=d,
        control=controlSpATS(maxit=500,tolerance=1e-6,monitoring=0)))[['elapsed']]
    results <- rbind(results,data.frame(language='R',random=gr,run=run,seconds=elapsed,iterations=m$niterations))
    cat(sprintf('n=%d random=%s run=%d seconds=%.6f iterations=%d\n',nrow(d),gr,run,elapsed,m$niterations))
    if(length(args)>=2 && run==repeats) {
      write.csv(data.frame(fitted=m$fitted),file.path(args[2],paste0('r_fitted_',gr,'.csv')),row.names=FALSE)
    }
    rm(m)
  }
}
if(length(args)>=2) write.csv(results,file.path(args[2],'r_timings.csv'),row.names=FALSE)
