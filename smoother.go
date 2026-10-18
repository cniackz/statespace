package statespace

import (
	"fmt"
	"math"
)

// smootherRcond is the relative eigenvalue floor for the correlation-scaled
// pseudo-inverse of a singular one-step predicted covariance. The cutoff is on
// the correlation matrix, whose eigenvalues are near one, so a direction is
// dropped only when it is collinear or has zero variance (a deterministic state
// component), never merely because its variance is small on an absolute scale.
const smootherRcond = 1e-12

// invertPredicted inverts a one-step predicted covariance for the smoother gain.
// The fast path is the ordinary inverse, accepted only when it is finite and its
// residual P inv - I is small, so a covariance that is merely ill-conditioned
// (mixed state units, a real but small-variance direction) keeps its full
// inverse and the common case pays no eigendecomposition. When the ordinary
// inverse fails or is inaccurate, as for a singular covariance (a deterministic
// state component from a rank-deficient Q), it falls back to the
// correlation-scaled pseudo-inverse, which drops only the deterministic
// directions and corrects the rest.
func invertPredicted(p Matrix) Matrix {
	if inv, ok := matInv(p); ok && inv.isFinite() && invAccurate(p, inv) {
		return inv
	}
	return correlationPseudoInverse(p, smootherRcond)
}

// invAccurate reports whether inv is a good inverse of p, by the max absolute
// entry of the residual p inv - I. The identity is dimensionless, so an absolute
// bound is scale-free; it is loose because its only job is to reject a grossly
// wrong inverse from a singular or near-singular p, not to grade a good one.
func invAccurate(p, inv Matrix) bool {
	r := matMul(p, inv)
	n := r.rows
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(r.At(i, j)-want) > 1e-6 {
				return false
			}
		}
	}
	return true
}

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
// covariance. A degenerate model that passes Validate but cannot be run in
// double precision, such as a variance near the overflow limit or a subnormal
// variance whose reciprocal scaling overflows, returns an error wrapping
// ErrNotFinite rather than a silent non-finite estimate. References: Rauch, Tung
// and Striebel 1965; Anderson and Moore 1979; Shumway and Stoffer 1982 for the
// lag-one covariance.
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
		predInv := invertPredicted(fs.predP[k+1])
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
		predInv := invertPredicted(fs.predP[t-1])
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
		if !vfinite(sx[k]) || !sp[k].isFinite() {
			return SmoothResult{}, fmt.Errorf("%w: smoother produced a non-finite estimate at step %d", ErrNotFinite, k)
		}
		res.Smoothed[k] = Gaussian{Mean: sx[k], Cov: sp[k]}
	}
	return res, nil
}
