// Package statespace is a small, dependency-free toolkit for the linear
// Gaussian state-space model of arbitrary dimension:
//
//	x_k = F x_{k-1} + B u_k + w_k,   w_k ~ N(0, Q)
//	z_k = H x_k + v_k,               v_k ~ N(0, R)
//
// with x_0 ~ N(Mu0, P0). It provides a numerically stable filter (the UD, or
// modified-Cholesky, factorization of Bierman and Thornton), the
// Rauch-Tung-Striebel fixed-interval smoother, and maximum-likelihood parameter
// estimation by the EM algorithm for linear dynamical systems.
//
// The mathematics is standard and the sources are cited on the types and
// functions that implement them: Kalman (1960); Rauch, Tung and Striebel (1965);
// Bierman, Factorization Methods for Discrete Sequential Estimation (1977);
// Thornton (1976); Dempster, Laird and Rubin (1977); Shumway and Stoffer (1982);
// Ghahramani and Hinton (1996); Anderson and Moore, Optimal Filtering (1979);
// Grewal and Andrews, Kalman Filtering: Theory and Practice (2015); Simon,
// Optimal State Estimation (2006); Durbin and Koopman (2012); Sarkka, Bayesian
// Filtering and Smoothing (2013).
//
// Cost: the filter, smoother and each EM iteration are dense and O(n^3) per time
// step in the state dimension n, O(T n^3) over a series of length T. The library
// is meant for moderate n.
//
// A nil observation in a sequence is a complete gap: the step runs the time
// update and skips the measurement update, and it adds nothing to the
// log-likelihood. A partial gap, where only some components of z are known, is
// not inferred from NaN entries; it is requested explicitly through a mask, and
// a NaN in an observation that is not a declared gap is invalid input, not a gap.
//
// No type here is safe for concurrent use without external synchronization.
package statespace

import "math"

// Vector is a dense real vector. A nil Vector in an observation sequence is a
// complete gap (see the package overview).
type Vector []float64

// Matrix is a dense, row-major real matrix. Build one with Zeros, Ident, Diag or
// FromRows; its contents are not mutated by the estimators, which copy what they
// keep.
type Matrix struct {
	rows, cols int
	data       []float64
}

// Zeros returns an r by c matrix of zeros. It panics only on a non-positive
// dimension, which is a programming error, not a data error.
func Zeros(r, c int) Matrix {
	if r <= 0 || c <= 0 {
		panic("statespace: matrix dimensions must be positive")
	}
	return Matrix{rows: r, cols: c, data: make([]float64, r*c)}
}

// Ident returns the n by n identity matrix.
func Ident(n int) Matrix {
	m := Zeros(n, n)
	for i := 0; i < n; i++ {
		m.data[i*n+i] = 1
	}
	return m
}

// Diag returns a square matrix with d on the diagonal and zeros elsewhere.
func Diag(d ...float64) Matrix {
	n := len(d)
	m := Zeros(n, n)
	for i := 0; i < n; i++ {
		m.data[i*n+i] = d[i]
	}
	return m
}

// FromRows returns a matrix with the given rows, which must all have the same
// length. It panics on a ragged or empty input, a programming error.
func FromRows(rows ...[]float64) Matrix {
	if len(rows) == 0 || len(rows[0]) == 0 {
		panic("statespace: FromRows needs at least one non-empty row")
	}
	c := len(rows[0])
	m := Zeros(len(rows), c)
	for i, row := range rows {
		if len(row) != c {
			panic("statespace: FromRows got rows of unequal length")
		}
		copy(m.data[i*c:(i+1)*c], row)
	}
	return m
}

// Rows returns the row count.
func (m Matrix) Rows() int { return m.rows }

// Cols returns the column count.
func (m Matrix) Cols() int { return m.cols }

// At returns the entry at row i, column j.
func (m Matrix) At(i, j int) float64 { return m.data[i*m.cols+j] }

// set writes the entry at row i, column j (internal; the public Matrix is
// immutable to callers).
func (m Matrix) set(i, j int, v float64) { m.data[i*m.cols+j] = v }

// clone returns an independent copy.
func (m Matrix) clone() Matrix {
	c := Matrix{rows: m.rows, cols: m.cols, data: make([]float64, len(m.data))}
	copy(c.data, m.data)
	return c
}

// isFinite reports whether every entry is finite.
func (m Matrix) isFinite() bool {
	for _, v := range m.data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// isSquare reports whether the matrix is square.
func (m Matrix) isSquare() bool { return m.rows == m.cols }

// isSymmetric reports whether the matrix is square and symmetric within a
// tolerance relative to its scale. The scale is the largest absolute entry, so a
// matrix whose entries are all tiny is still held to a relative tolerance: an
// absolute floor would call a matrix symmetric only because it is small.
func (m Matrix) isSymmetric(tol float64) bool {
	if !m.isSquare() {
		return false
	}
	var scale float64
	for _, v := range m.data {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	for i := 0; i < m.rows; i++ {
		for j := i + 1; j < m.cols; j++ {
			d := m.At(i, j) - m.At(j, i)
			if math.Abs(d) > tol*scale {
				return false
			}
		}
	}
	return true
}

// vfinite reports whether every entry of v is finite.
func vfinite(v Vector) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}
