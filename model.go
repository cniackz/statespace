package statespace

import (
	"errors"
	"fmt"
)

// symTol is the relative tolerance for the symmetry check on Q, R and P0, and
// psdTol the relative floor for their smallest eigenvalue. They are loose enough
// to accept a matrix a caller built by hand or from data, and tight enough to
// reject one that is meaningfully asymmetric or indefinite.
const (
	symTol = 1e-9
	psdTol = 1e-9
)

// ErrInvalidModel is returned by Validate, and by Filter, Smooth, NewKF and
// Learn, when a Model is malformed: a non-square or non-finite matrix, a
// dimension that does not match the others, a non-symmetric or non-positive
// semidefinite Q, R or P0, or a non-positive-definite R. Use errors.Is to test
// for it.
var ErrInvalidModel = errors.New("statespace: invalid model")

// ErrInvalidInput is returned by the estimators when an observation or control
// input is the wrong length, or carries a non-finite value in a position that is
// not a declared gap. Use errors.Is to test for it.
var ErrInvalidInput = errors.New("statespace: invalid input")

// ErrNotFinite is returned when a step would make the state or covariance
// non-finite; the estimator is left unchanged, as in a rejected observation. Use
// errors.Is to test for it.
var ErrNotFinite = errors.New("statespace: non-finite result")

// Model is a time-invariant linear Gaussian state-space model:
//
//	x_k = F x_{k-1} + B u_k + w_k,   w_k ~ N(0, Q)
//	z_k = H x_k + v_k,               v_k ~ N(0, R)
//
// with the prior x_0 ~ N(Mu0, P0). F and Q are n by n, H is m by n, R is m by m,
// Mu0 has length n and P0 is n by n. B is optional: a zero Matrix (the default)
// means no control input, otherwise B is n by p and each control input has
// length p. Q, R and P0 must be symmetric; Q and P0 must be positive
// semidefinite and R positive definite.
type Model struct {
	F   Matrix
	Q   Matrix
	H   Matrix
	R   Matrix
	Mu0 Vector
	P0  Matrix
	B   Matrix
}

// Gaussian is a mean vector and covariance matrix.
type Gaussian struct {
	Mean Vector
	Cov  Matrix
}

// stateDim returns n, the state dimension.
func (m Model) stateDim() int { return m.F.rows }

// obsDim returns the measurement dimension.
func (m Model) obsDim() int { return m.H.rows }

// ctrlDim returns the control dimension, or 0 when there is no B.
func (m Model) ctrlDim() int {
	if m.B.rows == 0 {
		return 0
	}
	return m.B.cols
}

// hasControl reports whether the model has a control-input matrix.
func (m Model) hasControl() bool { return m.B.rows != 0 }

// Validate reports whether the model is well formed. It returns an error
// wrapping ErrInvalidModel on the first problem it finds, and nil otherwise.
// Filter, Smooth, LogLikelihood, NewKF and Learn all call it first.
func (m Model) Validate() error {
	n := m.F.rows
	if n == 0 {
		return fmt.Errorf("%w: F is empty", ErrInvalidModel)
	}
	if !m.F.isSquare() {
		return fmt.Errorf("%w: F is %dx%d, not square", ErrInvalidModel, m.F.rows, m.F.cols)
	}
	if !m.Q.isSquare() || m.Q.rows != n {
		return fmt.Errorf("%w: Q must be %dx%d", ErrInvalidModel, n, n)
	}
	if m.H.rows == 0 || m.H.cols != n {
		return fmt.Errorf("%w: H must have %d columns and at least one row", ErrInvalidModel, n)
	}
	mm := m.H.rows
	if !m.R.isSquare() || m.R.rows != mm {
		return fmt.Errorf("%w: R must be %dx%d", ErrInvalidModel, mm, mm)
	}
	if !m.P0.isSquare() || m.P0.rows != n {
		return fmt.Errorf("%w: P0 must be %dx%d", ErrInvalidModel, n, n)
	}
	if len(m.Mu0) != n {
		return fmt.Errorf("%w: Mu0 must have length %d", ErrInvalidModel, n)
	}
	if m.hasControl() && m.B.rows != n {
		return fmt.Errorf("%w: B must have %d rows", ErrInvalidModel, n)
	}
	for _, mx := range []struct {
		name string
		mat  Matrix
	}{{"F", m.F}, {"Q", m.Q}, {"H", m.H}, {"R", m.R}, {"P0", m.P0}} {
		if !mx.mat.isFinite() {
			return fmt.Errorf("%w: %s has a non-finite entry", ErrInvalidModel, mx.name)
		}
	}
	if m.hasControl() && !m.B.isFinite() {
		return fmt.Errorf("%w: B has a non-finite entry", ErrInvalidModel)
	}
	if !vfinite(m.Mu0) {
		return fmt.Errorf("%w: Mu0 has a non-finite entry", ErrInvalidModel)
	}
	if !m.Q.isSymmetric(symTol) {
		return fmt.Errorf("%w: Q is not symmetric", ErrInvalidModel)
	}
	if !m.R.isSymmetric(symTol) {
		return fmt.Errorf("%w: R is not symmetric", ErrInvalidModel)
	}
	if !m.P0.isSymmetric(symTol) {
		return fmt.Errorf("%w: P0 is not symmetric", ErrInvalidModel)
	}
	if e := minEigSym(m.Q); e < -psdTol*(1+matScale(m.Q)) {
		return fmt.Errorf("%w: Q is not positive semidefinite (min eigenvalue %g)", ErrInvalidModel, e)
	}
	if e := minEigSym(m.P0); e < -psdTol*(1+matScale(m.P0)) {
		return fmt.Errorf("%w: P0 is not positive semidefinite (min eigenvalue %g)", ErrInvalidModel, e)
	}
	if _, ok := choleskyLower(m.R); !ok {
		return fmt.Errorf("%w: R is not positive definite", ErrInvalidModel)
	}
	return nil
}

// matScale returns a rough magnitude of a matrix (its largest absolute diagonal
// entry, or 1), used to scale the PSD tolerance.
func matScale(m Matrix) float64 {
	var s float64
	n := m.rows
	if m.cols < n {
		n = m.cols
	}
	for i := 0; i < n; i++ {
		if v := abs(m.At(i, i)); v > s {
			s = v
		}
	}
	if s == 0 {
		return 1
	}
	return s
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
