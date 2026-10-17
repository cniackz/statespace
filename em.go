package statespace

import "fmt"

// EMResult is the outcome of Learn: the fitted model, the log-likelihood at the
// start of each iteration (non-decreasing), the number of iterations run, and
// whether the log-likelihood increase fell below the tolerance.
type EMResult struct {
	Model     Model
	LogLik    []float64
	Iters     int
	Converged bool
}

// EMOption configures Learn.
type EMOption func(*emConfig)

type emConfig struct {
	maxIter int
	tol     float64
	fixed   map[string]bool
}

// WithMaxIterations caps the number of EM iterations (default 100).
func WithMaxIterations(n int) EMOption {
	return func(c *emConfig) {
		if n > 0 {
			c.maxIter = n
		}
	}
}

// WithTolerance sets the stopping tolerance on the per-iteration increase in
// log-likelihood (default 1e-6). EM stops when the increase is below it.
func WithTolerance(tol float64) EMOption {
	return func(c *emConfig) {
		if tol >= 0 {
			c.tol = tol
		}
	}
}

// WithFixed holds the named matrices fixed at their initial values during
// learning. Names are "F", "Q", "H", "R", "Mu0", "P0"; unknown names are
// ignored. Fix H, for example, when the observation model is known and only the
// dynamics are to be learned.
func WithFixed(names ...string) EMOption {
	return func(c *emConfig) {
		for _, n := range names {
			c.fixed[n] = true
		}
	}
}

// Learn fits F, Q, H, R, Mu0 and P0 by the EM algorithm, starting from init and
// using the observations zs (a nil entry is a gap). EM converges to a local
// optimum of the likelihood and is sensitive to the initial model: start from a
// reasonable init, and compare the returned log-likelihood trace across restarts
// if the fit is poor. The scale of the state and the sign of a state component
// are identifiable only up to the usual ambiguities of the linear Gaussian
// model, so a learned F and Q may differ from a data-generating pair that an H
// absorbs; the likelihood, not the individual matrices, is what EM maximizes.
// References: Dempster, Laird and Rubin 1977; Shumway and Stoffer 1982;
// Ghahramani and Hinton 1996.
//
// Learn does not estimate a control model: a model with a B is rejected with an
// error wrapping ErrInvalidModel. It validates the model it is about to return
// and reports an error wrapping ErrNotFinite if a divergent fit left it invalid,
// so it never returns a non-finite model with a nil error.
func Learn(zs []Vector, init Model, opts ...EMOption) (EMResult, error) {
	cfg := emConfig{maxIter: 100, tol: 1e-6, fixed: map[string]bool{}}
	for _, o := range opts {
		o(&cfg)
	}
	if len(zs) < 2 {
		return EMResult{}, fmt.Errorf("%w: EM needs at least two steps", ErrInvalidInput)
	}
	if err := init.Validate(); err != nil {
		return EMResult{}, err
	}
	if init.hasControl() {
		return EMResult{}, fmt.Errorf("%w: Learn does not estimate control models; remove B (Filter and Smooth do take control inputs)", ErrInvalidModel)
	}
	cur := cloneModel(init)
	var trace []float64
	converged := false
	iters := 0
	for iter := 0; iter < cfg.maxIter; iter++ {
		fs, err := cur.runFilter(zs, nil)
		if err != nil {
			return EMResult{}, fmt.Errorf("EM iteration %d: %w", iter, err)
		}
		sm, err := cur.smoothFrom(fs, zs)
		if err != nil {
			return EMResult{}, fmt.Errorf("EM iteration %d: %w", iter, err)
		}
		trace = append(trace, fs.loglik)
		iters = iter + 1
		if iter > 0 {
			inc := trace[iter] - trace[iter-1]
			if inc < cfg.tol {
				converged = true
				break
			}
		}
		cur = mStep(cur, zs, sm, cfg.fixed)
	}
	// The last M-step runs after the last likelihood evaluation, so validate the
	// model before returning it: a divergent fit (from extreme observations, say)
	// must surface as an error, never as a non-finite model with a nil error.
	if err := cur.Validate(); err != nil {
		return EMResult{}, fmt.Errorf("%w: EM did not reach a valid model (%v)", ErrNotFinite, err)
	}
	return EMResult{Model: cur, LogLik: trace, Iters: iters, Converged: converged}, nil
}

// mStep returns the model that maximizes the expected complete-data
// log-likelihood given the smoothed statistics, holding the fixed matrices.
func mStep(cur Model, zs []Vector, sm SmoothResult, fixed map[string]bool) Model {
	n := cur.stateDim()
	mm := cur.obsDim()
	t := len(zs)
	sx := make([]Vector, t)
	for k := range sm.Smoothed {
		sx[k] = sm.Smoothed[k].Mean
	}
	exx := make([]Matrix, t)
	for k := 0; k < t; k++ {
		exx[k] = add(sm.Smoothed[k].Cov, outer(sx[k], sx[k]))
	}
	exx1 := make([]Matrix, t) // exx1[k] = E[x_k x_{k-1}^T], k>=1
	for k := 1; k < t; k++ {
		exx1[k] = add(sm.LagOneCov[k], outer(sx[k], sx[k-1]))
	}
	out := cloneModel(cur)

	// Transition: F and Q from the T-1 transitions.
	sumNum := Zeros(n, n)
	sumDen := Zeros(n, n)
	for k := 1; k < t; k++ {
		sumNum = add(sumNum, exx1[k])
		sumDen = add(sumDen, exx[k-1])
	}
	fUsed := cur.F
	if !fixed["F"] {
		if denInv, ok := matInv(sumDen); ok {
			fUsed = matMul(sumNum, denInv)
			out.F = fUsed
		}
	}
	if !fixed["Q"] {
		acc := Zeros(n, n)
		for k := 1; k < t; k++ {
			// Exx[k] - F Exx1[k]^T - Exx1[k] F^T + F Exx[k-1] F^T.
			fe1t := matMul(fUsed, transpose(exx1[k]))
			e1ft := matMul(exx1[k], transpose(fUsed))
			fekft := matMul(matMul(fUsed, exx[k-1]), transpose(fUsed))
			acc = add(acc, add(sub(sub(exx[k], fe1t), e1ft), fekft))
		}
		q := scaleMat(symmetrize(acc), 1/float64(t-1))
		out.Q = liftPSD(q, psdTol*(1+matScale(q)))
	}

	// Observation: H and R over the observed steps.
	var nObs int
	sumHNum := Zeros(mm, n)
	sumHDen := Zeros(n, n)
	for k := 0; k < t; k++ {
		if zs[k] == nil {
			continue
		}
		nObs++
		sumHNum = add(sumHNum, outer(zs[k], sx[k]))
		sumHDen = add(sumHDen, exx[k])
	}
	hUsed := cur.H
	if !fixed["H"] && nObs > 0 {
		if denInv, ok := matInv(sumHDen); ok {
			hUsed = matMul(sumHNum, denInv)
			out.H = hUsed
		}
	}
	if !fixed["R"] && nObs > 0 {
		acc := Zeros(mm, mm)
		for k := 0; k < t; k++ {
			if zs[k] == nil {
				continue
			}
			// z z^T - H E[x] z^T - z E[x]^T H^T + H Exx[k] H^T.
			zz := outer(zs[k], zs[k])
			hxz := matMul(hUsed, outer(sx[k], zs[k]))
			zxh := matMul(outer(zs[k], sx[k]), transpose(hUsed))
			hexh := matMul(matMul(hUsed, exx[k]), transpose(hUsed))
			acc = add(acc, add(sub(sub(zz, hxz), zxh), hexh))
		}
		r := scaleMat(symmetrize(acc), 1/float64(nObs))
		out.R = liftPSD(r, (psdTol+1e-12)*(1+matScale(r)))
	}

	// Prior.
	if !fixed["Mu0"] {
		out.Mu0 = append(Vector(nil), sx[0]...)
	}
	if !fixed["P0"] {
		p0 := symmetrize(sm.Smoothed[0].Cov)
		out.P0 = liftPSD(p0, psdTol*(1+matScale(p0)))
	}
	return out
}

// outer returns a b^T.
func outer(a, b Vector) Matrix {
	m := Zeros(len(a), len(b))
	for i := range a {
		for j := range b {
			m.set(i, j, a[i]*b[j])
		}
	}
	return m
}

// scaleMat returns s*m.
func scaleMat(m Matrix, s float64) Matrix {
	out := m.clone()
	for i := range out.data {
		out.data[i] *= s
	}
	return out
}

// liftPSD adds (target - minEig)*I to m when its smallest eigenvalue is below
// target, so a covariance estimate that a finite-precision M-step nudged just
// under positive semidefinite is lifted back to pass validation. The nudge is
// tiny for a well-conditioned estimate.
func liftPSD(m Matrix, target float64) Matrix {
	e := minEigSym(m)
	if e >= target {
		return m
	}
	lift := target - e
	out := m.clone()
	for i := 0; i < out.rows; i++ {
		out.set(i, i, out.At(i, i)+lift)
	}
	return out
}

// cloneModel returns an independent copy of a model.
func cloneModel(m Model) Model {
	c := Model{
		F:   m.F.clone(),
		Q:   m.Q.clone(),
		H:   m.H.clone(),
		R:   m.R.clone(),
		Mu0: append(Vector(nil), m.Mu0...),
		P0:  m.P0.clone(),
	}
	if m.hasControl() {
		c.B = m.B.clone()
	}
	return c
}
