library(reshape2)
calcGramMtx <- function(mtx){
  n <- dim(mtx)[1]
  f <- function(x,y) (x-y)%*%(x-y)
  B <- outer(1:n,1:n,Vectorize(function(i,j) f(mtx[i,],mtx[j,])))+0.00001
  diag(B) <- 0
  return(B)
}

latGP.mdl.code <- "

  functions {
    matrix gp_exp_quad_cov(vector[] X, real Alpha, real Bandwidth, real jitter) {
      int dim_x = size(X);
      matrix[dim_x, dim_x] K_x = cov_exp_quad(X, Alpha, Bandwidth);
      for (n in 1:dim_x) K_x[n,n] = K_x[n,n] + jitter;
      //matrix[dim_x, dim_x] L_K_x = cholesky_decompose(K_x);
      return K_x;
    }

    vector gp_pred_rng(vector[] x_pred, vector[] xs_pred, vector Y, vector[] x_is, vector[] xs_is,
                       real alpha, real bwX, real bwS, real sigma) {
      
      int N_pred = size(x_pred); //size of training input
      int N = rows(Y); //size of training data
      vector[N_pred] f_pred; //predicted RE from GP

      {
        //Compute gram matrix of training set but with new jitter from sigma
        matrix[N,N] K_xx = gp_exp_quad_cov(x_is, alpha, bwX, square(sigma)); //gram matrix for covariates
        matrix[N,N] K_ss = gp_exp_quad_cov(xs_is, 1, bwS ,0); //gram matrix for spatial locations
        matrix[N,N] K_xsxs = K_xx .* K_ss; //Combined gram matrix
        
        //Compute gram matrix of testing set  with training set 
        matrix[N,N_pred] K_xx_pred = cov_exp_quad(x_is, x_pred, alpha, bwX);
        matrix[N,N_pred] K_ss_pred = cov_exp_quad(xs_is, xs_pred, 1, bwS);
        matrix[N,N_pred] K_xsxs_pred = K_xx_pred .* K_ss_pred; //Combined gram matrix

        //Compute gram matrix of testing set
        matrix[N_pred,N_pred] K_xx_2pred = cov_exp_quad(x_pred, alpha, bwX);
        matrix[N_pred,N_pred] K_ss_2pred = cov_exp_quad(xs_pred, 1, bwS);
        matrix[N_pred,N_pred] K_xsxs_2pred = K_xx_2pred .* K_ss_2pred; //Combined gram matrix

        //Compute cholesky decomposition of K_xsxs
        matrix[N,N] L_Sigma = cholesky_decompose(K_xsxs);

        //Compute mean of MVN for new predictions
        vector[N] Y_div_L = mdivide_left_tri_low(L_Sigma, Y);
        vector[N] K_div_Y = mdivide_right_tri_low(Y_div_L',L_Sigma)';
        vector[N_pred] f_pred_mu = (K_xsxs_pred' * K_div_Y);
        
        //Compute variance of MVN for new predictions
        matrix[N,N_pred] v_pred = mdivide_left_tri_low(L_Sigma, K_xsxs_pred);
        matrix[N_pred,N_pred] f_pred_cov = K_xsxs_2pred - v_pred' * v_pred;

        //Add jitter, compute predictions
        matrix[N_pred,N_pred] nugget = diag_matrix(rep_vector(1e-12,N_pred));
        f_pred = multi_normal_rng(f_pred_mu, f_pred_cov + nugget);
      }
    return f_pred;
    }
  }

  data {
    int<lower=1> N; //number of training points
    int<lower=1> N_pred; //number of test points
    int<lower=1> Mx; //number of non-spatial covariates
    int<lower=1> Ms; //number of spatial covariates
    vector[N] y; //training observations
    vector[Mx] x[N];   //training covariates
    vector[Ms] xs[N];  //training spatial locations
    vector[Mx] x_pred[N_pred]; //testing covariates
    vector[Ms] xs_pred[N_pred]; //testing covariates
  }

  parameters {
    real<lower=0> bw_Kx; //bandwidth of kernel for covariates
    real<lower=0> bw_Ks; //bandwidth of kernel for spatial locations
    real<lower=0> alpha; // marginal variance of kernel
    real<lower=0> sigma; // jitter
    vector[N] eta; //for non-centered parameterization
  }

  transformed parameters {
    vector[N] f; //realizations of RE
    {
      matrix[N,N] L; //cholesky deposition of gram matrix
      
      //Compute gram matrix
      matrix[N,N] Kx = gp_exp_quad_cov(x,alpha,bw_Kx,1e-12); //gram matrix for covariates
      matrix[N,N] Ks = gp_exp_quad_cov(xs,1,bw_Ks,0); //gram matrix for spatial locations
      matrix[N,N] K = Kx .* Ks; //Combined gram matrix
      
      //Compute realizations (f) of RE from kernel
      L = cholesky_decompose(K); 
      f = L * eta;
    }
  }

  model {
    bw_Kx ~ gamma(0.5, 0.5); //prior for bandwidth
    bw_Ks ~ gamma(2, 20); //prior for bandwidth
    alpha ~ normal(0, 1); //prior for marginal variance
    sigma ~ normal(0, 1); //prior for jitter
    eta ~ normal(0, 1);   //prior for transformation variable
    y ~ normal(f, sigma); //data generation process for observations
  }

  //generated quantities {
  //  vector[N_pred] f_pred; //RE for predictions
  //  vector[N_pred] y_pred; //predictions
  //  f_pred = gp_pred_rng(x_pred, xs_pred, y, x, xs, alpha, bw_Kx, bw_Ks, sigma); //Expected RE for y_pred
  //  for (n in 1:N_pred) y_pred[n] = normal_rng(f_pred[n], sigma);   //Realized RE for y_pred
  //}
"




dfHRM <- read.csv('/media/pato/DATA/Dev/features_config_id_78.csv')
dfSng <- read.csv('/media/pato/DATA/Dev/y_gps_config_78.csv')

###Use SVD for dimension reduction
eof.fit <- function(X, p) {
              s <- svd(X)
              s$u[, 1:p] %*% diag(s$d[1:p]) %*% t(s$v[, 1:p])
           }
eof <- function(X, p) {
          svd(X)$u[, 1:p]
       }
eof.summary <- function(X, per = c(25, 50, 95), col = c(1:3)) {
                  s <- svd(X)
                  svd.var <- s$d^2/sum(s$d^2) * 100
                  svd.cum.var <- cumsum(s$d^2/sum(s$d^2)) * 100
                  svd.per <- (1:ncol(X))[sapply(per, function(p) {min(which(svd.cum.var >= p))})]
                  svd.col <- cumsum(s$d[col]^2/sum(s$d^2)) * 100
                  
                  list(var = svd.var, cum.var = svd.cum.var, per.exp = svd.per,
                       per = per, col.exp = svd.col, col = col)
                }
lvis <- as.matrix(dfHRM[,c(1:512)])
X <- scale(lvis[, 3:ncol(lvis)], scale = FALSE)
eof.var <- eof.summary(X, per = c(50, 90, 95, 99), col = c(1:512))
par(mfrow = c(1, 2), mar = c(4, 6, 4, 2))
plot(eof.var[[1]], ylim = c(0, 100), cex.lab = 2, xlab = "SVD column",
      ylab = "Percent variance explained", pch = 19, cex = 0.2)
abline(v = eof.var[[3]], col = rainbow(length(eof.var[[3]])))
legend("topright", lty = 1, legend = paste("Column ", eof.var[[3]],
         ": ", eof.var[[4]], "%", sep = ""), col = rainbow(length(eof.var[[3]])),bty = "n")
plot(eof.var[[2]], ylim = c(0, 100), cex.lab = 2, xlab = "SVD column",ylab = "Cumulative percent variance explained", pch = 19, cex = 0.2)
abline(v = eof.var[[3]], col = rainbow(length(eof.var[[3]])))
legend("bottomright", lty = 1, legend = paste("Column ", eof.var[[3]],
        ": ", eof.var[[4]], "%", sep = ""), col = rainbow(length(eof.var[[3]])),bty = "n")

###Choose appropriate number of dimensions & create dataset
nVar <- 180
dfPCA <- cbind(dfHRM[,c('i','j')],eof(lvis,180))
dfSng <- as.data.frame(data.table::as.data.table(dfSng)[,list(y= mean(y)),c('i','j','lon','lat')])
dfHRM <- merge(dfSng,dfPCA,on=c('i','j'))[,c(3:(dim(dfPCA)[2]+3))]
#iDups <- which(duplicated(dfHRM[,c('lat')]))
#dfHRM$lat[iDups] <- jitter(dfHRM$lat[iDups])
#iDups <- which(duplicated(dfHRM[,c('lon')]))
#dfHRM$lon[iDups] <- jitter(dfHRM$lon[iDups])

#formula for ridge regression model
fmla <- as.formula(paste('y~',paste(colnames(dfHRM)[c(4:253)],sep='',collapse='+'),sep=''))

library(rstan)
library(spBayes)
library(glmnet)
library(caret)

#create train/test split
folds <- createFolds(dfHRM$y,k=5)
dfPrf <- data.frame(fold=numeric(),
                    mdl=character(),
                    rmse=numeric(),
                    mape=numeric(),
                    rsq=numeric())

jX <- c(4:(dim(dfHRM)[2]))
jS <- c('lat','lon')
X <- cbind(rep(1,dim(dfHRM)[1]),as.matrix(dfHRM[,jX]))

for (k in c(1:5)){
  
  iTst <- folds[[k]]
  iTrn <- c(1:dim(dfHRM)[1])[-iTst]
  
  ###train ridge regression model
  ridge.mdl  <- cv.glmnet(x=X[iTrn,], y=dfHRM$y[iTrn], alpha=0, lambda=10^seq(3,-2,by=-.1))
  lambda     <- ridge.mdl$lambda.min
  ridge.mdl  <- glmnet(x=X[iTrn,], y=dfHRM$y[iTrn], alpha=0, lambda=lambda)
  beta.prior <- unclass(coefficients(ridge.mdl))@x
  
  #Compute residuals <- train Gaussian Process on residuals
  rsd.spatial  <- dfHRM$y[iTrn]-predict(ridge.mdl,newx=X[iTrn,])
  
  #Create data input for STAN
  model.dat <- list(N  = length(iTrn) , N_pred = length(iTst), 
                    Mx = dim(X)[2] , Ms = 2,
                    y = as.vector(rsd.spatial),
                    x = X[iTrn,], xs = as.matrix(dfHRM[iTrn,c('lat','lon')]),
                    x_pred = X[iTst,],
                    xs_pred = as.matrix(dfHRM[i,c('lat','lon')]))
                    #control=list(adapt_delta=0.9,max_treedepth=12))
  # run model
  rstan_options(auto_write = TRUE)
  options(mc.cores = 4)
  stan.fit <- stan(model_code=latGP.mdl.code,
                   model_name="Latent GeoSpatial Gaussian_Process",
                   data=model.dat,
                   iter=1000,warmup=250,
                   chains=4,thin=1,seed=101)
  
  #extract results
  stan.rslt <- extract(stan.fit)
  bw_Kx <- mean(stan.rslt$bw_Kx)
  bw_Ks <- mean(stan.rslt$bw_Ks)
  alpha <- median(stan.rslt$alpha)
  sigma <- median(stan.rslt$sigma)

  #Compute Kernel
  Dx <- calcGramMtx(X)
  Ds <- calcGramMtx(as.matrix(dfHRM[,c('lat','lon')]))
  Kxs <- alpha^2*exp(-Dx/bw_Kx-Ds/bw_Ks)
  
  #Compute mean and variance K∗  θ  (Kθ + σ   2   I)  −1  y
  pred.mu <- Kxs[iTst,iTrn] %*% MASS::ginv(Kxs[iTrn,iTrn]+diag(length(iTrn))*sigma^2) %*% rsd.spatial
  pred.mu <- pred.mu+predict(ridge.mdl,newx=X[iTst,])
  
  # and plot
  #ggplot(cbind(dfHRM[-i,c('lat','lon')],rsd=rsd.vario), aes(x = lat, y = lon, color=s0))+geom_point()+scale_colour_gradient2()
  #
  #vario.mdl  <- variofit(variog(coords=as.matrix(dfHRM[-i,c('lat','lon')]),data=rsd.vario,option='smooth'),
  #                       cov.pars=c(var(rsd.vario),1),
  #                       cov.model='exponential')
  
  ###Train Gaussian Process model
  #Try Exponential Kernel
  starting <- list(#"beta"    = beta.prior,
                   "phi"      = 3/10,
                   "sigma.sq" = 2,
                   "tau.sq"   = 2)
  tuning <- list("phi"=0.5, 
                 "sigma.sq"=0.5, "tau.sq"=0.5)
  priors <- list(#beta.Norm = list(rep(0,length(beta.prior)), diag(1000,length(beta.prior))),
                 phi.Unif = c(0.1,10), 
                 sigma.sq.IG = c(2, 2), 
                 tau.sq.IG = c(2, 2))
  sp.mdl.exp <- spLM(s0~1,data=as.data.frame(rsd.vario),coords=as.matrix(dfHRM[-i,c('lat','lon')]),cov.model='exponential',
                     starting = starting,tuning = tuning, priors = priors,n.samples=5000)
  #mdl.exp.rslt <- spRecover(sp.mdl.exp,start=1001)
  
  #Try Matern Kernel
  starting <- list(#"beta"    = beta.prior,
    "nu"       = 2,
    "phi"      = 3/10,
    "sigma.sq" = 2,
    "tau.sq"   = 2)
  tuning <- list("phi"=0.5, "nu"=0.5,
                 "sigma.sq"=0.5, "tau.sq"=0.5)
  priors <- list(#beta.Norm = list(rep(0,length(beta.prior)), diag(1000,length(beta.prior))),
    nu.Unif  = c(0.5,5),
    phi.Unif = c(0.1,10), 
    sigma.sq.IG = c(2, 2), 
    tau.sq.IG = c(2, 2))
  sp.mdl.mtn <- spLM(s0~1,data=as.data.frame(rsd.vario),coords=as.matrix(dfHRM[-i,c('lat','lon')]),cov.model='matern',
                     starting = starting,tuning = tuning, priors = priors,n.samples=5000)
  #mdl.mtn.rslt <- spRecover(sp.mdl.exp,start=1001)
  
  ###Perform predictions
  pred.exp <- spPredict(sp.mdl.exp, start=1001, thin=4, 
                        pred.coords=as.matrix(dfHRM[i,c('lat','lon')]),pred.covars=as.matrix(rep(1,length(i))))
  summ.exp <- t(apply(pred.exp$p.y.predictive.samples, 1, function(x) {c(mean(x),var(x),quantile(x, prob = c(0.05, 0.5, 0.95)))}))
  colnames(summ.exp) <- c('Avg','Vrnc','5%','50%','95%')
  
  pred.mtn <- spPredict(sp.mdl.mtn, start=1001, thin=4, 
                        pred.coords=as.matrix(dfHRM[i,c('lat','lon')]),pred.covars=as.matrix(rep(1,length(i))))
  summ.mtn <- t(apply(pred.mtn$p.y.predictive.samples, 1, function(x) {c(mean(x),var(x),quantile(x, prob = c(0.05, 0.5, 0.95)))}))
  colnames(summ.mtn) <- c('Avg','Vrnc','5%','50%','95%')
  
  ###Compute RMSE
  dfPrd <- cbind(y=dfHRM$y[i],
                 ridge = predict(ridge.mdl,newx=cbind(rep(1,length(i)),as.matrix(dfHRM[i,c(4:(dim(dfHRM)[2]))]))))
  dfPrd <- as.data.frame(cbind(dfPrd,
                         expGP = dfPrd[,2]+summ.exp[,1],
                         mtnGP = dfPrd[,2]+summ.mtn[,1]))
  colnames(dfPrd) <- c('y','yRdg','yExp','yMtn')
  dfPrd$yGP <- summ.exp[,1]
  
  
  #perf.rdg <- c('rmse'=ModelMetrics::rmse(dfPrd$y,dfPrd$yRdg),
  #              'rsq' =1-(sum((dfPrd$y-dfPrd$yRdg)^2)/sum((dfPrd$y-mean(dfPrd$y))^2)),
  #              'mape'=mean(abs((dfPrd$y-dfPrd$yRdg)/dfPrd$y)))
  #perf.exp <- c('rmse'=ModelMetrics::rmse(dfPrd$y,dfPrd$yExp),
  #              'rsq' =1-(sum((dfPrd$y-dfPrd$yExp)^2)/sum((dfPrd$y-mean(dfPrd$y))^2)),
  #              'mape'=mean(abs((dfPrd$y-dfPrd$yExp)/dfPrd$y)))
  #perf.mtn <- c('rmse'=ModelMetrics::rmse(dfPrd$y,dfPrd$yMtn),
  #              'rsq' =1-(sum((dfPrd$y-dfPrd$yMtn)^2)/sum((dfPrd$y-mean(dfPrd$y))^2)),
  #              'mape'=mean(abs((dfPrd$y-dfPrd$yMtn)/dfPrd$y)))
    
  dfPrf <- rbind(dfPrf,data.frame(fold=k,mdl='rdg',
                                  rmse=ModelMetrics::rmse(dfPrd$y,dfPrd$yRdg),
                                  mape=mean(abs((dfPrd$y-dfPrd$yRdg)/dfPrd$y)),
                                  rsq =1-(sum((dfPrd$y-dfPrd$yRdg)^2)/sum((dfPrd$y-mean(dfPrd$y))^2))))
  dfPrf <- rbind(dfPrf,data.frame(fold=k,mdl='exp',
                                  rmse=ModelMetrics::rmse(dfHRM$y[iTst],pred.mu),
                                  mape=mean(abs((dfHRM$y[iTst]-pred.mu)/dfHRM$y[iTst])),
                                  rsq =1-(sum((dfHRM$y[iTst]-pred.mu)^2)/sum((dfHRM$y[iTst]-mean(dfHRM$y[iTst]))^2))))
  dfPrf <- rbind(dfPrf,data.frame(fold=k,mdl='mtn',
                                  rmse=ModelMetrics::rmse(dfPrd$y,dfPrd$yMtn),
                                  mape=mean(abs((dfPrd$y-dfPrd$yMtn)/dfPrd$y)),
                                  rsq =1-(sum((dfPrd$y-dfPrd$yMtn)^2)/sum((dfPrd$y-mean(dfPrd$y))^2))))
  dfPrf <- rbind(dfPrf,data.frame(fold=k,mdl='GProd',
                                  rmse=ModelMetrics::rmse(dfPrd$y,dfPrd$yGP),
                                  mape=mean(abs((dfPrd$y-dfPrd$yGP)/dfPrd$y)),
                                  rsq =1-(sum((dfPrd$y-dfPrd$yGP)^2)/sum((dfPrd$y-mean(dfPrd$y))^2))))
}

ggmap(get_map(location = c(lon=mean(dfHRM$lon), lat=mean(dfHRM$lat)), zoom=1,maptype="terrain",scale = 2))
  +     geom_point(data = cbind(dfHRM[-i,c('lat','lon')],rsd=rsd.vario),aes(x = lat, y = lon, fill = s0)) +
  +     guides(fill=FALSE, alpha=FALSE, size=FALSE)