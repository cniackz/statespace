package statespace

// smootherRcond is the relative eigenvalue floor for the pseudo-inverse of the
// one-step predicted covariance in the smoother. A direction with variance below
// this fraction of the largest is treated as deterministic and gets no smoothing
// correction, so a model with a deterministic state component (a singular
// predicted covariance) is smoothed instead of rejected.
const smootherRcond = 1e-12

// SmoothResult holds the fixed-interval smoothed estimate x_k|T at every step,
// the total log-likelihood of the observations, and the lag-one smoothed
// covariances Cov(x_k, x_{k-1} | T) that EM needs. LagOneCov has the same length
// as the series; LagOneCov[0] is the zero matrix (there is no x_{-1}).
type SmoothResult struct {
	Smoothed      []Gaussian
	LogLikelihood float64
	LagOneCov     []Matrix
}

// Smooth runs the forward filter and the Rauch-Tung-Striebel backward recursion,
// returning the smoothed estimate at every step. us is nil or one control input
// per step, as in Filter. A deterministic state direction (a one-step predicted
// covariance that is singular, from a rank-deficient Q) gets no smoothing
// correction rather than an error, through a pseudo-inverse of the predicted
// covariance. References: Rauch, Tung and Striebel 1965; Anderson and Moore
// 1979; Shumway and Stoffer 1982 for the lag-one covariance.
func (m Model) Smooth(zs []Vector, us []Vector) (SmoothResult, error) {
	fs, err := m.runFilter(zs, us)
	if err != nil {
		return SmoothResult{}, err
	}
	return m.smoothFrom(fs, zs)
}

// smoothFrom runs the backward recursion on a completed forward pass. It is the
// E-step core shared by Smooth and Learn.
func (m Model) smoothFrom(fs filterState, zs []Vector) (SmoothResult, error) {
	n := m.stateDim()
	t := len(zs)
	ft := transpose(m.F)
	sx := make([]Vector, t)
	sp := make([]Matrix, t)
	gains := make([]Matrix, t) // gains[k] = J_k for k in 0..t-2
	sx[t-1] = fs.filtX[t-1]
	sp[t-1] = fs.filtP[t-1]
	for k := t - 2; k >= 0; k-- {
		predInv := pseudoInverseSym(fs.predP[k+1], smootherRcond)
		j := matMul(matMul(fs.filtP[k], ft), predInv)
		gains[k] = j
		dx := make(Vector, n)
		for i := 0; i < n; i++ {
			dx[i] = sx[k+1][i] - fs.predX[k+1][i]
		}
		jdx := matVec(j, dx)
		xk := make(Vector, n)
		for i := 0; i < n; i++ {
			xk[i] = fs.filtX[k][i] + jdx[i]
		}
		sx[k] = xk
		dP := sub(sp[k+1], fs.predP[k+1])
		sp[k] = symmetrize(add(fs.filtP[k], matMul(matMul(j, dP), transpose(j))))
	}
	lag := make([]Matrix, t)
	lag[0] = Zeros(n, n)
	if t >= 2 {
		// Lag-one initialization: P_{t-1,t-2|T} = (I - K H) F P_{t-2|t-2}, with
		// (I - K H) = P_{t-1|t-1} (P_{t-1|t-2})^{-1}, which holds whether or not
		// the last step carried an observation (Ghahramani and Hinton 1996).
		predInv := pseudoInverseSym(fs.predP[t-1], smootherRcond)
		ikh := matMul(fs.filtP[t-1], predInv)
		lag[t-1] = matMul(matMul(ikh, m.F), fs.filtP[t-2])
		// lag[o] = filtP[o] J_{o-1}^T + J_o (lag[o+1] - F filtP[o]) J_{o-1}^T,
		// for o from t-2 down to 1; J_k = gains[k] is defined for k in 0..t-2.
		for o := t - 2; o >= 1; o-- {
			jo1t := transpose(gains[o-1])
			fp := matMul(m.F, fs.filtP[o])
			term := matMul(matMul(gains[o], sub(lag[o+1], fp)), jo1t)
			lag[o] = add(matMul(fs.filtP[o], jo1t), term)
		}
	}
	res := SmoothResult{
		Smoothed:      make([]Gaussian, t),
		LogLikelihood: fs.loglik,
		LagOneCov:     lag,
	}
	for k := 0; k < t; k++ {
		res.Smoothed[k] = Gaussian{Mean: sx[k], Cov: sp[k]}
	}
	return res, nil
}
