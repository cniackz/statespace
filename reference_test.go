package statespace

import (
	"math"
	"math/rand"
	"testing"
)

// This file holds an independent moment-form reference filter with the
// Joseph-form covariance update, used only to cross-check the UD filter. The two
// are algebraically the same Kalman filter; agreement to a tight tolerance on
// random systems is the evidence that the UD implementation is correct.

// refFilter runs the textbook Kalman filter in moment form with the Joseph
// covariance update. A nil observation is a gap. It returns the filtered means,
// filtered covariances and total log-likelihood.
func refFilter(m Model, zs []Vector, us []Vector) ([]Vector, []Matrix, float64) {
	n := m.F.rows
	mm := m.H.rows
	x := append(Vector(nil), m.Mu0...)
	p := m.P0.clone()
	fx := make([]Vector, len(zs))
	fp := make([]Matrix, len(zs))
	var ll float64
	for step := range zs {
		if step > 0 {
			x = matVec(m.F, x)
			if m.hasControl() && us != nil {
				bu := matVec(m.B, us[step])
				for i := range x {
					x[i] += bu[i]
				}
			}
			p = add(matMul(matMul(m.F, p), transpose(m.F)), m.Q)
		}
		if zs[step] != nil {
			// S = H P H^T + R
			hp := matMul(m.H, p)
			s := add(matMul(hp, transpose(m.H)), m.R)
			sInv := inv(s)
			// K = P H^T S^{-1}
			k := matMul(matMul(p, transpose(m.H)), sInv)
			hx := matVec(m.H, x)
			e := make(Vector, mm)
			for i := range e {
				e[i] = zs[step][i] - hx[i]
			}
			ke := matVec(k, e)
			for i := range x {
				x[i] += ke[i]
			}
			// Joseph: P = (I-KH) P (I-KH)^T + K R K^T
			ikh := sub(Ident(n), matMul(k, m.H))
			p = add(matMul(matMul(ikh, p), transpose(ikh)), matMul(matMul(k, m.R), transpose(k)))
			ll += -0.5 * (float64(mm)*ln2pi + math.Log(det(s)) + quad(e, sInv))
		}
		fx[step] = append(Vector(nil), x...)
		fp[step] = p.clone()
	}
	return fx, fp, ll
}

// quad returns e^T A e.
func quad(e Vector, a Matrix) float64 {
	ae := matVec(a, e)
	var s float64
	for i := range e {
		s += e[i] * ae[i]
	}
	return s
}

// inv returns the inverse of a small square matrix by Gauss-Jordan elimination
// with partial pivoting. Test-only.
func inv(a Matrix) Matrix {
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
		for j := 0; j < 2*n; j++ {
			t := aug.At(col, j)
			aug.set(col, j, aug.At(piv, j))
			aug.set(piv, j, t)
		}
		d := aug.At(col, col)
		for j := 0; j < 2*n; j++ {
			aug.set(col, j, aug.At(col, j)/d)
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
	return out
}

// det returns the determinant of a small square matrix by LU with partial
// pivoting. Test-only.
func det(a Matrix) float64 {
	n := a.rows
	lu := a.clone()
	d := 1.0
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(lu.At(r, col)) > math.Abs(lu.At(piv, col)) {
				piv = r
			}
		}
		if piv != col {
			for j := 0; j < n; j++ {
				t := lu.At(col, j)
				lu.set(col, j, lu.At(piv, j))
				lu.set(piv, j, t)
			}
			d = -d
		}
		d *= lu.At(col, col)
		if lu.At(col, col) == 0 {
			return 0
		}
		for r := col + 1; r < n; r++ {
			f := lu.At(r, col) / lu.At(col, col)
			for j := col; j < n; j++ {
				lu.set(r, j, lu.At(r, j)-f*lu.At(col, j))
			}
		}
	}
	return d
}

// randSPD returns a random n by n symmetric positive definite matrix.
func randSPD(rng *rand.Rand, n int, scale float64) Matrix {
	a := Zeros(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a.set(i, j, rng.NormFloat64())
		}
	}
	p := matMul(a, transpose(a))
	for i := 0; i < n; i++ {
		p.set(i, i, p.At(i, i)+scale)
	}
	return p
}

// randStable returns a random n by n matrix with spectral radius below 1, scaled
// down from a random matrix so the simulated series stays bounded.
func randStable(rng *rand.Rand, n int) Matrix {
	a := Zeros(n, n)
	var maxAbs float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := rng.NormFloat64()
			a.set(i, j, v)
			if math.Abs(v) > maxAbs {
				maxAbs = math.Abs(v)
			}
		}
	}
	s := 0.6 / (maxAbs * float64(n))
	for i := range a.data {
		a.data[i] *= s
	}
	return a
}

func randModel(rng *rand.Rand, n, mm int) Model {
	return Model{
		F:   randStable(rng, n),
		Q:   randSPD(rng, n, 0.1),
		H:   randMat(rng, mm, n),
		R:   randSPD(rng, mm, 0.2),
		Mu0: randVec(rng, n),
		P0:  randSPD(rng, n, 1.0),
	}
}

func randMat(rng *rand.Rand, r, c int) Matrix {
	m := Zeros(r, c)
	for i := range m.data {
		m.data[i] = rng.NormFloat64()
	}
	return m
}

func randVec(rng *rand.Rand, n int) Vector {
	v := make(Vector, n)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	return v
}

// simulate draws an observation sequence from the model, inserting gaps at the
// given step indices.
func simulate(rng *rand.Rand, m Model, t int, gaps map[int]bool) []Vector {
	n := m.F.rows
	lq, _ := choleskyLower(m.Q)
	lr, _ := choleskyLower(m.R)
	lp, _ := choleskyLower(m.P0)
	x := make(Vector, n)
	x0 := matVec(lp, randVec(rng, n))
	for i := range x {
		x[i] = m.Mu0[i] + x0[i]
	}
	zs := make([]Vector, t)
	for step := 0; step < t; step++ {
		if step > 0 {
			x = matVec(m.F, x)
			w := matVec(lq, randVec(rng, n))
			for i := range x {
				x[i] += w[i]
			}
		}
		if gaps[step] {
			zs[step] = nil
			continue
		}
		hx := matVec(m.H, x)
		v := matVec(lr, randVec(rng, m.H.rows))
		z := make(Vector, m.H.rows)
		for i := range z {
			z[i] = hx[i] + v[i]
		}
		zs[step] = z
	}
	return zs
}

func maxVecDiff(a, b Vector) float64 {
	var d float64
	for i := range a {
		if x := math.Abs(a[i] - b[i]); x > d {
			d = x
		}
	}
	return d
}

func maxMatDiff(a, b Matrix) float64 {
	var d float64
	for i := range a.data {
		if x := math.Abs(a.data[i] - b.data[i]); x > d {
			d = x
		}
	}
	return d
}

// TestUDMatchesJoseph is claim 1 and claim 8: the UD filter reproduces the
// textbook Joseph-form filter in mean, covariance and log-likelihood, including
// sequences with gaps, to a tight relative tolerance over many random systems.
func TestUDMatchesJoseph(t *testing.T) {
	rng := rand.New(rand.NewSource(20261017))
	for trial := 0; trial < 200; trial++ {
		n := 1 + rng.Intn(4)
		mm := 1 + rng.Intn(3)
		m := randModel(rng, n, mm)
		if err := m.Validate(); err != nil {
			t.Fatalf("trial %d: model invalid: %v", trial, err)
		}
		tt := 10 + rng.Intn(20)
		gaps := map[int]bool{}
		for step := 1; step < tt; step++ {
			if rng.Float64() < 0.15 {
				gaps[step] = true
			}
		}
		zs := simulate(rng, m, tt, gaps)
		res, err := m.Filter(zs, nil)
		if err != nil {
			t.Fatalf("trial %d: Filter: %v", trial, err)
		}
		fx, fp, ll := refFilter(m, zs, nil)
		for step := 0; step < tt; step++ {
			if d := maxVecDiff(res.Filtered[step].Mean, fx[step]); d > 1e-8 {
				t.Fatalf("trial %d step %d: mean diff %g", trial, step, d)
			}
			if d := maxMatDiff(res.Filtered[step].Cov, fp[step]); d > 1e-8 {
				t.Fatalf("trial %d step %d: cov diff %g", trial, step, d)
			}
		}
		if d := math.Abs(res.LogLikelihood - ll); d > 1e-7*(1+math.Abs(ll)) {
			t.Fatalf("trial %d: loglik UD %g vs Joseph %g (diff %g)", trial, res.LogLikelihood, ll, d)
		}
	}
}

// TestCovStaysPSD is claim 2: after every step the reconstructed covariance is
// symmetric and positive semidefinite.
func TestCovStaysPSD(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 100; trial++ {
		n := 1 + rng.Intn(4)
		mm := 1 + rng.Intn(3)
		m := randModel(rng, n, mm)
		tt := 15
		zs := simulate(rng, m, tt, nil)
		res, err := m.Filter(zs, nil)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		for step := 0; step < tt; step++ {
			p := res.Filtered[step].Cov
			if !p.isSymmetric(1e-9) {
				t.Fatalf("trial %d step %d: cov not symmetric", trial, step)
			}
			if e := minEigSym(p); e < -1e-9*(1+matScale(p)) {
				t.Fatalf("trial %d step %d: cov not PSD, min eig %g", trial, step, e)
			}
		}
	}
}
