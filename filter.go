package statespace

import (
	"fmt"
	"math"
)

const ln2pi = 1.8378770664093454835606594728112 // ln(2*pi)

// FilterResult holds, per time step, the one-step predicted estimate (before the
// observation) and the filtered estimate (after it), and the total
// log-likelihood of the observed steps. A gap contributes a predicted and a
// filtered estimate that are equal and adds nothing to the log-likelihood.
type FilterResult struct {
	Predicted     []Gaussian
	Filtered      []Gaussian
	LogLikelihood float64
}

// KF is an online Kalman filter for a Model, carrying the state estimate and its
// covariance in UD-factored form. Build it with NewKF, drive it with Predict and
// Update, and read it with State and Cov. It is not safe for concurrent use.
type KF struct {
	n, m int
	f    Matrix
	b    Matrix
	h    Matrix
	r    Matrix
	uq   Matrix // UD factors of Q
	dq   Vector
	x    Vector
	u    Matrix // unit upper triangular; P = U diag(d) U^T
	d    Vector
}

// NewKF returns an online filter initialized at the model prior x_0 ~ N(Mu0,P0).
// It returns an error wrapping ErrInvalidModel if the model is malformed.
func NewKF(model Model) (*KF, error) {
	if err := model.Validate(); err != nil {
		return nil, err
	}
	uq, dq := udFactor(model.Q)
	u, d := udFactor(model.P0)
	x := make(Vector, len(model.Mu0))
	copy(x, model.Mu0)
	return &KF{
		n:  model.stateDim(),
		m:  model.obsDim(),
		f:  model.F.clone(),
		b:  cloneOrEmpty(model.B),
		h:  model.H.clone(),
		r:  model.R.clone(),
		uq: uq,
		dq: dq,
		x:  x,
		u:  u,
		d:  d,
	}, nil
}

func cloneOrEmpty(m Matrix) Matrix {
	if m.rows == 0 {
		return Matrix{}
	}
	return m.clone()
}

// State returns a copy of the current state estimate.
func (k *KF) State() Vector {
	out := make(Vector, len(k.x))
	copy(out, k.x)
	return out
}

// Cov reconstructs and returns the current state covariance P = U diag(d) U^T.
func (k *KF) Cov() Matrix { return udReconstruct(k.u, k.d) }

// Clone returns an independent copy of the filter.
func (k *KF) Clone() *KF {
	c := *k
	c.f = k.f.clone()
	c.b = cloneOrEmpty(k.b)
	c.h = k.h.clone()
	c.r = k.r.clone()
	c.uq = k.uq.clone()
	c.dq = append(Vector(nil), k.dq...)
	c.x = append(Vector(nil), k.x...)
	c.u = k.u.clone()
	c.d = append(Vector(nil), k.d...)
	return &c
}

// Predict advances the filter one time step: x <- F x + B u and
// P <- F P F^T + Q. The control u is used only if the model has a B; pass nil
// otherwise. It computes the next state in scratch and commits only if it is
// finite, so a step that would make the state non-finite returns ErrNotFinite
// and leaves the filter unchanged. A wrong-length control returns ErrInvalidInput.
func (k *KF) Predict(u Vector) error {
	newX := matVec(k.f, k.x)
	if k.b.rows != 0 {
		if len(u) != k.b.cols {
			return fmt.Errorf("%w: control has length %d, want %d", ErrInvalidInput, len(u), k.b.cols)
		}
		if !vfinite(u) {
			return fmt.Errorf("%w: control has a non-finite entry", ErrInvalidInput)
		}
		bu := matVec(k.b, u)
		for i := range newX {
			newX[i] += bu[i]
		}
	}
	newU, newD := thorntonUpdate(k.u, k.d, k.f, k.uq, k.dq)
	if !vfinite(newX) || !newU.isFinite() || !vfinite(newD) {
		return ErrNotFinite
	}
	k.x = newX
	k.u = newU
	k.d = newD
	return nil
}

// Update folds one observation into the estimate and returns the step's
// contribution to the log-likelihood. A nil z is a complete gap: nothing changes
// and the contribution is zero. A z of the wrong length, or one carrying a
// non-finite value, returns ErrInvalidInput and leaves the filter unchanged. If
// the update would make the estimate non-finite it returns ErrNotFinite and
// leaves the filter unchanged.
func (k *KF) Update(z Vector) (float64, error) {
	if z == nil {
		return 0, nil
	}
	if len(z) != k.m {
		return 0, fmt.Errorf("%w: observation has length %d, want %d", ErrInvalidInput, len(z), k.m)
	}
	if !vfinite(z) {
		return 0, fmt.Errorf("%w: observation has a non-finite entry (use nil for a gap)", ErrInvalidInput)
	}
	return k.foldObs(k.h, k.r, z)
}

// UpdateMasked folds a partial observation: mask[i] true means component i of z
// is observed, false means it is missing. A z of the wrong length, a mask of the
// wrong length, or a non-finite value in an observed position returns
// ErrInvalidInput. With every entry false the step is a complete gap. Missing
// components are dropped together with their rows of H and rows and columns of R,
// which is the exact marginal treatment.
func (k *KF) UpdateMasked(z Vector, mask []bool) (float64, error) {
	if len(z) != k.m || len(mask) != k.m {
		return 0, fmt.Errorf("%w: observation and mask must have length %d", ErrInvalidInput, k.m)
	}
	var idx []int
	for i, ok := range mask {
		if ok {
			if math.IsNaN(z[i]) || math.IsInf(z[i], 0) {
				return 0, fmt.Errorf("%w: observed component %d is non-finite", ErrInvalidInput, i)
			}
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 {
		return 0, nil
	}
	hSub := Zeros(len(idx), k.n)
	rSub := Zeros(len(idx), len(idx))
	zSub := make(Vector, len(idx))
	for a, i := range idx {
		zSub[a] = z[i]
		for c := 0; c < k.n; c++ {
			hSub.set(a, c, k.h.At(i, c))
		}
		for b2, j := range idx {
			rSub.set(a, b2, k.r.At(i, j))
		}
	}
	return k.foldObs(hSub, rSub, zSub)
}

// foldObs whitens the observation with the Cholesky factor of r, then folds each
// whitened scalar component in with Bierman's update, accumulating the exact
// log-likelihood of z. It computes in scratch and commits only on success.
func (k *KF) foldObs(h, r Matrix, z Vector) (float64, error) {
	l, ok := choleskyLower(r)
	if !ok {
		return 0, fmt.Errorf("%w: measurement-noise block is not positive definite", ErrInvalidModel)
	}
	zw := solveLowerVec(l, z)
	hw := solveLowerMat(l, h)
	var logDetL float64
	for i := 0; i < l.rows; i++ {
		logDetL += math.Log(l.At(i, i))
	}
	// Work on copies so a non-finite result leaves the filter untouched.
	xs := append(Vector(nil), k.x...)
	us := k.u.clone()
	ds := append(Vector(nil), k.d...)
	var ll float64
	for c := 0; c < hw.rows; c++ {
		hRow := make(Vector, k.n)
		for j := 0; j < k.n; j++ {
			hRow[j] = hw.At(c, j)
		}
		innov, s, okUp := biermanUpdate(us, ds, xs, hRow, 1, zw[c])
		if !okUp {
			return 0, ErrNotFinite
		}
		ll += -0.5 * (ln2pi + math.Log(s) + innov*innov/s)
	}
	ll -= logDetL
	if !vfinite(xs) || !us.isFinite() || !vfinite(ds) || math.IsNaN(ll) || math.IsInf(ll, 0) {
		return 0, ErrNotFinite
	}
	k.x = xs
	k.u = us
	k.d = ds
	return ll, nil
}

// Filter runs the forward pass over the observations zs, with optional control
// inputs us (nil, or one per step; us[k] drives the transition into step k, so
// us[0] is unused). A nil entry in zs is a complete gap. It returns the
// per-step predicted and filtered estimates and the total log-likelihood.
func (m Model) Filter(zs []Vector, us []Vector) (FilterResult, error) {
	fs, err := m.runFilter(zs, us)
	if err != nil {
		return FilterResult{}, err
	}
	res := FilterResult{
		Predicted:     make([]Gaussian, len(zs)),
		Filtered:      make([]Gaussian, len(zs)),
		LogLikelihood: fs.loglik,
	}
	for kk := range zs {
		res.Predicted[kk] = Gaussian{Mean: fs.predX[kk], Cov: fs.predP[kk]}
		res.Filtered[kk] = Gaussian{Mean: fs.filtX[kk], Cov: fs.filtP[kk]}
	}
	return res, nil
}

// LogLikelihood returns the total log-likelihood of zs under the model.
func (m Model) LogLikelihood(zs []Vector, us []Vector) (float64, error) {
	fs, err := m.runFilter(zs, us)
	if err != nil {
		return 0, err
	}
	return fs.loglik, nil
}

// filterState is the moment-form record of a forward pass, kept for the smoother
// and EM.
type filterState struct {
	predX, filtX []Vector
	predP, filtP []Matrix
	loglik       float64
}

// runFilter is the shared forward pass. It stores, per step, the predicted and
// filtered mean and covariance in moment form.
func (m Model) runFilter(zs []Vector, us []Vector) (filterState, error) {
	if err := m.Validate(); err != nil {
		return filterState{}, err
	}
	if us != nil && len(us) != len(zs) {
		return filterState{}, fmt.Errorf("%w: us must be nil or the same length as zs", ErrInvalidInput)
	}
	k, err := NewKF(m)
	if err != nil {
		return filterState{}, err
	}
	t := len(zs)
	fs := filterState{
		predX: make([]Vector, t),
		filtX: make([]Vector, t),
		predP: make([]Matrix, t),
		filtP: make([]Matrix, t),
	}
	for step := 0; step < t; step++ {
		if step > 0 {
			var u Vector
			if us != nil {
				u = us[step]
			}
			if err := k.Predict(u); err != nil {
				return filterState{}, fmt.Errorf("step %d predict: %w", step, err)
			}
		}
		fs.predX[step] = k.State()
		fs.predP[step] = k.Cov()
		ll, err := k.Update(zs[step])
		if err != nil {
			return filterState{}, fmt.Errorf("step %d update: %w", step, err)
		}
		fs.loglik += ll
		fs.filtX[step] = k.State()
		fs.filtP[step] = k.Cov()
	}
	return fs, nil
}
