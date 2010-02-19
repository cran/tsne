tsne <-
function(X,initial_config = NULL, k=2, initial_dims=30, perplexity=30, max_iter = 1000, min_cost=0, epoch_callback=NULL,whiten=TRUE, epoch=100 ){


	if (class(X) == 'dist') { 
		n = attr(X,'Size')
		X = X/sum(X)
		}
	else 	{
		X = as.matrix(X)
		initial_dims = min(initial_dims,ncol(X))
		if (whiten) X<-.whiten(as.matrix(X),n.comp=initial_dims)
		n = dim(X)[1]
	}

	momentum = .5
	final_momentum = .8
	mom_switch_iter = 250

	epsilon = 500
	min_gain = .01

	P = .x2p(X,perplexity, 1e-5)$P

	eps = 2^(-52) # typical machine precision
	P[is.nan(P)]<-eps
	P = .5 * (P + t(P))
	P = P / sum(P)
	P[P < eps]<-eps
	P = P * 4
	if (!is.null(initial_config)) { 
		ydata = initial_config
	} else {
		ydata = matrix(rnorm(k * nrow(X)),nrow(X))
	}
	y_grads =  matrix(0,dim(ydata)[1],dim(ydata)[2])
	y_incs =  matrix(0,dim(ydata)[1],dim(ydata)[2])
	gains = matrix(1,dim(ydata)[1],dim(ydata)[2])
	
	for (iter in 1:max_iter){
		sum_ydata = apply(ydata^2, 1, sum)
		num =  1/(1 + sum_ydata +    sweep(-2 * ydata %*% t(ydata),2, -t(sum_ydata))) 
		diag(num)=0
		Q = num / sum(num)
		if (any(is.nan(num))) message ('NaN in grad. descent')
		Q[Q < eps] = eps
		stiffnesses = 4 * (P-Q) * num
		for (i in 1:n){
			y_grads[i,] = apply(sweep(-ydata, 2, -ydata[i,]) * stiffnesses[,i],2,sum)
		}
		
		gains = (gains + .2) * abs(sign(y_grads) != sign(y_incs)) 
				+ gains * .8 * abs(sign(y_grads) == sign(y_incs))		
		gains[gains < min_gain] = min_gain
		y_incs = momentum * y_incs - epsilon * (gains * y_grads)
		ydata = ydata + y_incs
		y_data = sweep(ydata,2,apply(ydata,2,mean))
		if (iter == mom_switch_iter) momentum = final_momentum
		
		if (iter == 100) P = P/4
		
		if (iter %% epoch == 0) { # epoch
			cost =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
			message("Epoch: Iteration #",iter," error is: ",cost)
			if (cost < min_cost) break
			if (!is.null(epoch_callback)) epoch_callback(ydata, P)
		}
	
		
	}
	r = {}
	r$ydata = ydata
	r$P = P
	r
	
}

