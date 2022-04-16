data {
    int<lower=0> K;
    real df[K];
    real mu[K];
}
parameters {
    real x[K];
}
model {
    for(k in 1:K) {
        target += normal_lpdf(x[k] | mu[k], df[k]);
    }
}