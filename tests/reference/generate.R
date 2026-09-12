# Run from repository root with the supplied SpATS 1.0-20 installed.
library(SpATS)
stopifnot(as.character(packageVersion('SpATS')) == '1.0.20')
set.seed(9122026)
d <- expand.grid(col=1:12, row=1:16)
d$geno <- factor(sample(rep(sprintf('G%02d',1:24),8)))
d$block <- factor(ceiling(d$row/4))
d$treatment <- factor(sample(rep(c('A','B'),96)))
d$population <- factor(ifelse(as.integer(d$geno)<=12, 'P1','P2'))
d$weight <- runif(nrow(d),0.6,1.8)
d$off <- 0.1*d$col
g <- rnorm(24,0,2)
d$yield <- 20 + g[d$geno] + 0.2*d$row + sin(d$col/2)*2 +
    cos(d$row/3)*1.5 + 0.1*d$col*d$row + (d$treatment=='B')*2 +
    rnorm(4,0,1)[d$block] + rnorm(nrow(d),0,0.8)
d$count <- rpois(nrow(d), exp(1+g[d$geno]/5 + sin(d$col/3)/3))
d$binary <- rbinom(nrow(d), 1, plogis(g[d$geno]/2 + sin(d$col/3)))
write.csv(d, 'tests/reference/field.csv', row.names=FALSE)
for (kind in c('PSANOVA','SAP','SAP.ANOVA')) for (gr in c(FALSE,TRUE)) {
    form <- if(kind=='PSANOVA') ~PSANOVA(col,row,nseg=c(6,8),nest.div=c(2,2)) else
        if(kind=='SAP') ~SAP(col,row,nseg=c(6,8),nest.div=c(2,2)) else
            ~SAP(col,row,nseg=c(6,8),nest.div=c(2,2),ANOVA=TRUE)
    model <- SpATS(response='yield',genotype='geno',genotype.as.random=gr,
        spatial=form,fixed=~treatment,random=~block,data=d,
        control=controlSpATS(tolerance=1e-9,maxit=1000,monitoring=0))
    tag <- paste0(kind,if(gr)'_random' else '_fixed')
    write.csv(data.frame(fitted=model$fitted,se=predict(model,newdata=d)$standard.errors),paste0('tests/reference/',tag,'_fitted.csv'),row.names=FALSE)
    write.csv(data.frame(name=names(model$var.comp),variance=model$var.comp,
        ed=tail(model$eff.dim,length(model$var.comp))),
        paste0('tests/reference/',tag,'_components.csv'),row.names=FALSE)
    write.csv(data.frame(psi=model$psi[1],objective=model$deviance,
        h2=if(gr)unname(model$eff.dim['geno']/model$dim.nom['geno']) else NA),
        paste0('tests/reference/',tag,'_stats.csv'),row.names=FALSE)
}
# Weighted offset case, missing responses, excluded plots, centered bases, populations, GLMMs.
for (case in c('weighted','missing','centered','populations','poisson','binomial')) {
    dd <- d
    if(case=='missing') {dd$yield[c(2,10,50)] <- NA; dd$weight[20] <- 0}
    model <- SpATS(response=if(case=='poisson')'count' else if(case=='binomial')'binary' else 'yield',
        genotype='geno',genotype.as.random=TRUE,
        geno.decomp=if(case=='populations')'population' else NULL,
        spatial=if(case=='centered')~PSANOVA(col,row,nseg=c(6,8),nest.div=2,center=TRUE) else
            ~PSANOVA(col,row,nseg=c(6,8),nest.div=2),
        fixed=~treatment, random=~block, data=dd,
        weights=if(case %in% c('weighted','missing'))dd$weight else NULL,
        offset=if(case=='weighted')dd$off else 0,
        family=if(case=='poisson')poisson() else if(case=='binomial')binomial() else gaussian(),
        control=controlSpATS(tolerance=1e-9,maxit=1000,monitoring=0))
    write.csv(data.frame(fitted=model$fitted,se=predict(model,newdata=dd)$standard.errors),paste0('tests/reference/',case,'_fitted.csv'),row.names=FALSE)
    write.csv(data.frame(name=names(model$var.comp),variance=model$var.comp,
        ed=tail(model$eff.dim,length(model$var.comp))),paste0('tests/reference/',case,'_components.csv'),row.names=FALSE)
    write.csv(data.frame(psi=model$psi[1],objective=model$deviance),paste0('tests/reference/',case,'_stats.csv'),row.names=FALSE)
}
# The real public wheat trial shipped by R SpATS (not simulated).
data(wheatdata)
write.csv(wheatdata,'pyspats/data/wheat.csv',row.names=FALSE)
wheatdata$R <- factor(wheatdata$row)
wheatdata$C <- factor(wheatdata$col)
for (gr in c(FALSE,TRUE)) {
  model <- SpATS(response='yield',genotype='geno',genotype.as.random=gr,
    spatial=~PSANOVA(col,row,nseg=c(10,10)),random=~R+C,data=wheatdata,
    control=controlSpATS(tolerance=1e-9,maxit=1000,monitoring=0))
  tag <- if(gr)'wheat_random' else 'wheat_fixed'
  write.csv(data.frame(fitted=model$fitted),paste0('tests/reference/',tag,'_fitted.csv'),row.names=FALSE)
  write.csv(data.frame(name=names(model$var.comp),variance=model$var.comp,
    ed=tail(model$eff.dim,length(model$var.comp))),paste0('tests/reference/',tag,'_components.csv'),row.names=FALSE)
  write.csv(data.frame(psi=model$psi[1],objective=model$deviance),paste0('tests/reference/',tag,'_stats.csv'),row.names=FALSE)
}
# Existing project sorghum data, including zero-response factor levels.
s <- read.csv('examples/sorghum_data.csv',check.names=FALSE,na.strings=c('', 'NA'))
# Explicitly align pandas' blank-ID handling and the Python complete-predictor policy.
s_valid <- complete.cases(s[,c('PINumber','Column','Row','Treatment','Block')])
s_n <- nrow(s)
s <- s[s_valid,]
s$PINumber <- factor(s$PINumber)
s$Treatment <- factor(s$Treatment)
s$Block <- factor(s$Block)
model <- SpATS(response='EstimatedPlotYield',genotype='PINumber',genotype.as.random=TRUE,
    spatial=~PSANOVA(Column,Row,nseg=c(10,10),nest.div=2),fixed=~Treatment,random=~Block,data=s,
    control=controlSpATS(tolerance=1e-7,maxit=1000,monitoring=0))
s_fitted <- rep(NA,s_n)
s_fitted[s_valid] <- model$fitted
write.csv(data.frame(fitted=s_fitted),'tests/reference/sorghum_fitted.csv',row.names=FALSE)
write.csv(data.frame(name=names(model$var.comp),variance=model$var.comp,
    ed=tail(model$eff.dim,length(model$var.comp))),'tests/reference/sorghum_components.csv',row.names=FALSE)
# Nondefault degrees and SAP penalty orders exercise tensor dimension/scaling.
for(case in c('unequal_degree','unequal_order')) {
  form <- if(case=='unequal_degree')~PSANOVA(col,row,nseg=c(6,8),nest.div=2,degree=c(2,4)) else
    ~SAP(col,row,nseg=c(6,8),nest.div=2,degree=c(2,3),pord=c(1,3))
  model <- SpATS(response='yield',genotype='geno',genotype.as.random=TRUE,
    spatial=form,fixed=~treatment,random=~block,data=d,
    control=controlSpATS(tolerance=1e-9,maxit=1000,monitoring=0))
  write.csv(data.frame(fitted=model$fitted,se=predict(model,newdata=d)$standard.errors),
    paste0('tests/reference/',case,'_fitted.csv'),row.names=FALSE)
  write.csv(data.frame(name=names(model$var.comp),variance=model$var.comp,
    ed=tail(model$eff.dim,length(model$var.comp))),paste0('tests/reference/',case,'_components.csv'),row.names=FALSE)
  write.csv(data.frame(psi=model$psi[1],objective=model$deviance),paste0('tests/reference/',case,'_stats.csv'),row.names=FALSE)
}
