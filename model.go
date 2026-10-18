package statespace

import (
	"errors"
	"fmt"
	"math"
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
	if err := covProblem(m.Q, "Q", false); err != nil {
		return err
	}
	if err := covProblem(m.R, "R", true); err != nil {
		return err
	}
	if err := covProblem(m.P0, "P0", false); err != nil {
		return err
	}
	return nil
}

// covProblem checks that a covariance matrix is symmetric and positive
// semidefinite (positive definite when requirePD), with every test relative to
// the scale of each component rather than to a single global magnitude. It works
// on the correlation matrix D^-1/2 m D^-1/2: a small indefinite or asymmetric
// block next to a large one stays visible, where a tolerance tied to the largest
// entry would hide it (the entries of a correlation matrix are near one). It
// returns an error wrapping ErrInvalidModel, naming the matrix, or nil.
func covProblem(m Matrix, name string, requirePD bool) error {
	n := m.rows
	var dmax float64
	for i := 0; i < n; i++ {
		d := m.At(i, i)
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return fmt.Errorf("%w: %s has a non-finite diagonal entry", ErrInvalidModel, name)
		}
		if a := math.Abs(d); a > dmax {
			dmax = a
		}
	}
	if dmax == 0 {
		if requirePD {
			return fmt.Errorf("%w: %s is not positive definite (zero diagonal)", ErrInvalidModel, name)
		}
		for _, v := range m.data {
			if v != 0 {
				return fmt.Errorf("%w: %s has a zero diagonal but a non-zero entry, not positive semidefinite", ErrInvalidModel, name)
			}
		}
		return nil
	}
	// A positive-(semi)definite matrix has a non-negative diagonal, each entry
	// judged against its own row (a negative variance is invalid however small it
	// is next to a larger component). A component has usable variance when 1/sqrt
	// of its diagonal is finite; a zero or subnormal diagonal is a deterministic
	// (zero-variance) direction, which a positive-definite matrix cannot have.
	// The per-component factor 1/sqrt(d_i) scales the correlation and the symmetry
	// test without ever forming d_i*d_j, which would overflow or underflow at
	// extreme scales.
	scale := make([]float64, n)
	rowScale := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if a := math.Abs(m.At(i, j)); a > rowScale[i] {
				rowScale[i] = a
			}
		}
	}
	for i := 0; i < n; i++ {
		d := m.At(i, i)
		if d < -psdTol*rowScale[i] {
			return fmt.Errorf("%w: %s has a negative diagonal entry, not positive semidefinite", ErrInvalidModel, name)
		}
		if s := 1 / math.Sqrt(d); d > 0 && !math.IsInf(s, 0) && !math.IsNaN(s) {
			scale[i] = s
		} else if requirePD {
			return fmt.Errorf("%w: %s is not positive definite (a diagonal entry is zero)", ErrInvalidModel, name)
		}
	}
	// Build the correlation matrix and check symmetry on its scale. A component
	// whose variance is zero must have zero covariance, judged against the other
	// component's standard deviation sqrt(d_j), not against the largest diagonal,
	// or a covariance tied to a tiny variance slips next to a large block.
	c := Ident(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if scale[i] != 0 && scale[j] != 0 {
				if math.Abs(m.At(i, j)-m.At(j, i))*scale[i]*scale[j] > symTol {
					return fmt.Errorf("%w: %s is not symmetric", ErrInvalidModel, name)
				}
				r := m.At(i, j) * scale[i] * scale[j]
				c.set(i, j, r)
				c.set(j, i, r)
			} else {
				ref := rowScale[i]
				if rowScale[j] > ref {
					ref = rowScale[j]
				}
				if scale[j] != 0 {
					ref = 1 / scale[j]
				} else if scale[i] != 0 {
					ref = 1 / scale[i]
				}
				if math.Abs(m.At(i, j)-m.At(j, i)) > symTol*ref {
					return fmt.Errorf("%w: %s is not symmetric", ErrInvalidModel, name)
				}
				if math.Abs(m.At(i, j)) > symTol*ref || math.Abs(m.At(j, i)) > symTol*ref {
					return fmt.Errorf("%w: %s has a zero-variance component with non-zero covariance, not positive semidefinite", ErrInvalidModel, name)
				}
			}
		}
	}
	mn, mx := eigExtremesSym(c)
	if math.IsNaN(mn) || math.IsInf(mn, 0) {
		return fmt.Errorf("%w: %s has a non-finite eigenvalue", ErrInvalidModel, name)
	}
	floor := mx
	if floor < 1 {
		floor = 1
	}
	if mn < -psdTol*floor {
		return fmt.Errorf("%w: %s is not positive semidefinite (correlation min eigenvalue %g)", ErrInvalidModel, name, mn)
	}
	if requirePD && mn <= psdTol*floor {
		return fmt.Errorf("%w: %s is not positive definite (correlation min eigenvalue %g)", ErrInvalidModel, name, mn)
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
