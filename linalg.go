package statespace

import "math"

// This file holds the dense linear-algebra primitives and the two
// factorization updates that make the filter numerically stable. Nothing here
// is exported: the public surface is Model, KF and the result types. The
// covariance is always carried as P = U D U^T, with U unit upper triangular and
// D a non-negative diagonal, so P never loses symmetry or positive
// semidefiniteness in finite precision (Bierman 1977; Thornton 1976).

// matMul returns a*b.
func matMul(a, b Matrix) Matrix {
	out := Zeros(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for k := 0; k < a.cols; k++ {
			aik := a.data[i*a.cols+k]
			if aik == 0 {
				continue
			}
			for j := 0; j < b.cols; j++ {
				out.data[i*b.cols+j] += aik * b.data[k*b.cols+j]
			}
		}
	}
	return out
}

// matVec returns a*v.
func matVec(a Matrix, v Vector) Vector {
	out := make(Vector, a.rows)
	for i := 0; i < a.rows; i++ {
		var s float64
		for j := 0; j < a.cols; j++ {
			s += a.data[i*a.cols+j] * v[j]
		}
		out[i] = s
	}
	return out
}

// transpose returns a^T.
func transpose(a Matrix) Matrix {
	out := Zeros(a.cols, a.rows)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			out.data[j*a.rows+i] = a.data[i*a.cols+j]
		}
	}
	return out
}

// addInto returns a+b (a and b same shape).
func add(a, b Matrix) Matrix {
	out := a.clone()
	for i := range out.data {
		out.data[i] += b.data[i]
	}
	return out
}

// sub returns a-b (a and b same shape).
func sub(a, b Matrix) Matrix {
	out := a.clone()
	for i := range out.data {
		out.data[i] -= b.data[i]
	}
	return out
}

// choleskyLower returns the lower-triangular L with L L^T = m and reports
// whether m was numerically positive definite. It is used to whiten a general
// measurement-noise matrix R, and to test a caller matrix for PSD.
func choleskyLower(m Matrix) (Matrix, bool) {
	n := m.rows
	if n != m.cols {
		return Matrix{}, false
	}
	l := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := m.At(i, j)
			for k := 0; k < j; k++ {
				s -= l.At(i, k) * l.At(j, k)
			}
			if i == j {
				if s <= 0 || math.IsNaN(s) || math.IsInf(s, 0) {
					return Matrix{}, false
				}
				l.set(i, j, math.Sqrt(s))
			} else {
				l.set(i, j, s/l.At(j, j))
			}
		}
	}
	return l, true
}

// solveLowerVec returns y with L y = b by forward substitution; L is lower
// triangular with nonzero diagonal.
func solveLowerVec(l Matrix, b Vector) Vector {
	n := l.rows
	y := make(Vector, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= l.At(i, k) * y[k]
		}
		y[i] = s / l.At(i, i)
	}
	return y
}

// solveLowerMat returns Y with L Y = B by forward substitution, column by
// column.
func solveLowerMat(l, b Matrix) Matrix {
	n := l.rows
	out := Zeros(n, b.cols)
	for c := 0; c < b.cols; c++ {
		for i := 0; i < n; i++ {
			s := b.At(i, c)
			for k := 0; k < i; k++ {
				s -= l.At(i, k) * out.At(k, c)
			}
			out.set(i, c, s/l.At(i, i))
		}
	}
	return out
}

// udFactor returns the UD factorization of a symmetric positive semidefinite
// matrix p: U unit upper triangular and d a non-negative diagonal with
// U diag(d) U^T = p. It is the construction of Bierman (1977), and it tolerates
// a singular p (a zero pivot gives a zero d entry and a zero column of U), which
// a plain Cholesky cannot. The input is assumed symmetric; callers pass a
// Validate-checked matrix.
func udFactor(p Matrix) (Matrix, Vector) {
	n := p.rows
	u := Ident(n)
	d := make(Vector, n)
	work := p.clone()
	for j := n - 1; j >= 0; j-- {
		dj := work.At(j, j)
		if dj < 0 {
			dj = 0
		}
		d[j] = dj
		if dj == 0 {
			// Zero pivot: the column of U stays zero (identity diagonal), and
			// the rows above are left as they are; nothing to subtract.
			continue
		}
		for i := 0; i <= j-1; i++ {
			u.set(i, j, work.At(i, j)/dj)
		}
		for k := 0; k <= j-1; k++ {
			for i := 0; i <= k; i++ {
				work.set(i, k, work.At(i, k)-u.At(i, j)*dj*u.At(k, j))
			}
		}
	}
	return u, d
}

// udReconstruct returns P = U diag(d) U^T.
func udReconstruct(u Matrix, d Vector) Matrix {
	n := len(d)
	p := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += u.At(i, k) * d[k] * u.At(j, k)
			}
			p.set(i, j, s)
		}
	}
	return p
}

// biermanUpdate folds one scalar observation z = h . x + noise(var r) into the
// state x and its UD factors U, D in place. It is Bierman's rank-one
// observational update (1977): it processes the measurement without ever forming
// P, keeping D non-negative. r must be positive. It returns the innovation
// z - h.x and the innovation variance (h P h^T + r); ok is false only if the
// innovation variance is not positive and finite, in which case nothing is
// mutated.
func biermanUpdate(u Matrix, d Vector, x Vector, h Vector, r, z float64) (innov, innovVar float64, ok bool) {
	n := len(d)
	// a = U^T h, so a_i = h_i + sum_{k<i} U[k][i] h[k].
	a := make(Vector, n)
	for i := 0; i < n; i++ {
		s := h[i]
		for k := 0; k < i; k++ {
			s += u.At(k, i) * h[k]
		}
		a[i] = s
	}
	// b_i = d_i a_i.
	b := make(Vector, n)
	for i := 0; i < n; i++ {
		b[i] = d[i] * a[i]
	}
	// Innovation variance S = r + sum_i a_i b_i = r + h P h^T.
	s := r
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	if s <= 0 || math.IsNaN(s) || math.IsInf(s, 0) {
		return 0, 0, false
	}
	var hx float64
	for i := 0; i < n; i++ {
		hx += h[i] * x[i]
	}
	dz := z - hx
	if math.IsNaN(dz) || math.IsInf(dz, 0) {
		return 0, 0, false
	}
	// Sequential update of U and D with the running gain accumulator k.
	dNew := make(Vector, n)
	uNew := u.clone()
	k := make(Vector, n)
	alpha := r + a[0]*b[0]
	if alpha <= 0 {
		return 0, 0, false
	}
	dNew[0] = d[0] * r / alpha
	k[0] = b[0]
	for j := 1; j < n; j++ {
		beta := alpha
		alpha += a[j] * b[j]
		if alpha <= 0 {
			return 0, 0, false
		}
		dNew[j] = d[j] * beta / alpha
		lambda := -a[j] / beta
		for i := 0; i < j; i++ {
			oldU := u.At(i, j)
			uNew.set(i, j, oldU+lambda*k[i])
			k[i] += b[j] * oldU
		}
		k[j] = b[j]
	}
	// Commit: x += (k / S) dz, D <- dNew, U <- uNew.
	for i := 0; i < n; i++ {
		x[i] += k[i] / alpha * dz
	}
	copy(d, dNew)
	copy(u.data, uNew.data)
	return dz, s, true
}

// thorntonUpdate propagates the UD factors through one time step
// x <- F x and P <- F P F^T + Q, with Q given by its own UD factors (uq, dq).
// It is Thornton's modified weighted Gram-Schmidt temporal update (1976), which
// keeps D non-negative. It returns the new unit-upper U and non-negative D.
func thorntonUpdate(u Matrix, d Vector, f, uq Matrix, dq Vector) (Matrix, Vector) {
	n := len(d)
	q := len(dq)
	// phiU = F U (n x n); the augmented row space is [phiU | uq], width n+q,
	// with weights [d | dq].
	phiU := matMul(f, u)
	g := uq.clone()
	w := make(Vector, n+q)
	copy(w, d)
	copy(w[n:], dq)
	uOut := Ident(n)
	dOut := make(Vector, n)
	// Row i of the augmented matrix: phiU row i (length n) then g row i (length q).
	for i := n - 1; i >= 0; i-- {
		// sigma = weighted squared norm of row i.
		var sigma float64
		for c := 0; c < n; c++ {
			sigma += phiU.At(i, c) * phiU.At(i, c) * w[c]
		}
		for c := 0; c < q; c++ {
			sigma += g.At(i, c) * g.At(i, c) * w[n+c]
		}
		dOut[i] = sigma
		if sigma <= 0 {
			// Degenerate row: leave column of U as identity, D entry zero.
			dOut[i] = 0
			continue
		}
		for kk := 0; kk < i; kk++ {
			// dot = weighted inner product of row kk and row i.
			var dot float64
			for c := 0; c < n; c++ {
				dot += phiU.At(kk, c) * w[c] * phiU.At(i, c)
			}
			for c := 0; c < q; c++ {
				dot += g.At(kk, c) * w[n+c] * g.At(i, c)
			}
			uOut.set(kk, i, dot/sigma)
			// Orthogonalize row kk against row i.
			coef := dot / sigma
			for c := 0; c < n; c++ {
				phiU.set(kk, c, phiU.At(kk, c)-coef*phiU.At(i, c))
			}
			for c := 0; c < q; c++ {
				g.set(kk, c, g.At(kk, c)-coef*g.At(i, c))
			}
		}
	}
	return uOut, dOut
}

// jacobiEigSym returns the eigenvalues and eigenvectors of a symmetric matrix by
// the cyclic Jacobi method. The k-th eigenvector is column k of the returned
// matrix, so m = V diag(eig) V^T. The matrix is normalized by its largest
// absolute entry before the sweeps, which makes the convergence cutoff relative
// to the scale of the matrix: a matrix whose entries are all near 1e-16 is
// rotated as thoroughly as one near 1, and its eigenvalues are not read off a
// diagonal that was never touched. n is moderate, and Jacobi is reliable for
// small symmetric matrices.
func jacobiEigSym(m Matrix) (Vector, Matrix) {
	n := m.rows
	var scale float64
	for _, v := range m.data {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		return make(Vector, n), Ident(n)
	}
	a := scaleMat(m, 1/scale)
	v := Ident(n)
	for sweep := 0; sweep < 100; sweep++ {
		var off float64
		for p := 0; p < n; p++ {
			for qq := p + 1; qq < n; qq++ {
				off += a.At(p, qq) * a.At(p, qq)
			}
		}
		if off <= 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for qq := p + 1; qq < n; qq++ {
				apq := a.At(p, qq)
				if apq == 0 {
					continue
				}
				theta := (a.At(qq, qq) - a.At(p, p)) / (2 * apq)
				t := sign(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta == 0 {
					t = 1
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for i := 0; i < n; i++ {
					aip := a.At(i, p)
					aiq := a.At(i, qq)
					a.set(i, p, c*aip-s*aiq)
					a.set(i, qq, s*aip+c*aiq)
				}
				for i := 0; i < n; i++ {
					api := a.At(p, i)
					aqi := a.At(qq, i)
					a.set(p, i, c*api-s*aqi)
					a.set(qq, i, s*api+c*aqi)
				}
				for i := 0; i < n; i++ {
					vip := v.At(i, p)
					viq := v.At(i, qq)
					v.set(i, p, c*vip-s*viq)
					v.set(i, qq, s*vip+c*viq)
				}
			}
		}
	}
	eig := make(Vector, n)
	for i := 0; i < n; i++ {
		eig[i] = a.At(i, i) * scale
	}
	return eig, v
}

// eigExtremesSym returns the smallest eigenvalue and the largest absolute
// eigenvalue of a symmetric matrix. The pair lets a positive-semidefinite test
// use a tolerance relative to the scale of the matrix, which is the whole point:
// an absolute floor would accept a matrix whose indefiniteness is small only
// because the matrix itself is small.
func eigExtremesSym(m Matrix) (minEig, maxAbs float64) {
	eig, _ := jacobiEigSym(m)
	minEig = eig[0]
	maxAbs = math.Abs(eig[0])
	for _, e := range eig[1:] {
		if e < minEig {
			minEig = e
		}
		if a := math.Abs(e); a > maxAbs {
			maxAbs = a
		}
	}
	return minEig, maxAbs
}

// minEigSym returns the smallest eigenvalue of a symmetric matrix.
func minEigSym(m Matrix) float64 {
	mn, _ := eigExtremesSym(m)
	return mn
}

// pseudoInverseSym returns the Moore-Penrose pseudo-inverse of a symmetric
// matrix, inverting only the eigen-directions whose eigenvalue magnitude exceeds
// reltol times the largest. A direction with zero variance (a deterministic
// state component) is left at zero, so the smoother applies no correction along
// it instead of failing on a singular predicted covariance. For a
// positive-definite matrix this equals the ordinary inverse.
func pseudoInverseSym(m Matrix, reltol float64) Matrix {
	eig, v := jacobiEigSym(m)
	n := len(eig)
	var maxAbs float64
	for _, e := range eig {
		if a := math.Abs(e); a > maxAbs {
			maxAbs = a
		}
	}
	cutoff := reltol * maxAbs
	d := make(Vector, n)
	for i, e := range eig {
		if math.Abs(e) > cutoff {
			d[i] = 1 / e
		}
	}
	out := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += v.At(i, k) * d[k] * v.At(j, k)
			}
			out.set(i, j, s)
		}
	}
	return out
}

func sign(x float64) float64 {
	if x < 0 {
		return -1
	}
	return 1
}

// matInv returns the inverse of a square matrix by Gauss-Jordan elimination with
// partial pivoting, and reports whether it was non-singular. It is used by the
// smoother and EM to invert small predicted covariances and sufficient-statistic
// sums.
func matInv(a Matrix) (Matrix, bool) {
	n := a.rows
	aug := Zeros(n, 2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug.set(i, j, a.At(i, j))
		}
		aug.set(i, n+i, 1)
	}
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(aug.At(r, col)) > math.Abs(aug.At(piv, col)) {
				piv = r
			}
		}
		if aug.At(piv, col) == 0 || math.IsNaN(aug.At(piv, col)) {
			return Matrix{}, false
		}
		for j := 0; j < 2*n; j++ {
			t := aug.At(col, j)
			aug.set(col, j, aug.At(piv, j))
			aug.set(piv, j, t)
		}
		dv := aug.At(col, col)
		for j := 0; j < 2*n; j++ {
			aug.set(col, j, aug.At(col, j)/dv)
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug.At(r, col)
			for j := 0; j < 2*n; j++ {
				aug.set(r, j, aug.At(r, j)-f*aug.At(col, j))
			}
		}
	}
	out := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out.set(i, j, aug.At(i, n+j))
		}
	}
	return out, true
}

// symmetrize returns (m + m^T)/2, used to clean tiny asymmetries from a
// covariance estimate before it is returned or re-factored.
func symmetrize(m Matrix) Matrix {
	n := m.rows
	out := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out.set(i, j, 0.5*(m.At(i, j)+m.At(j, i)))
		}
	}
	return out
}
