package statespace

import (
	"math"
	"math/rand"
	"testing"
)

// cvModel builds a constant-velocity tracker: state [position, velocity],
// F = [[1,dt],[0,1]], white-acceleration process noise of intensity q, position
// observed with measurement variance r.
func cvModel(dt, q, r float64) Model {
	return Model{
		F: FromRows([]float64{1, dt}, []float64{0, 1}),
		Q: FromRows(
			[]float64{q * dt * dt * dt / 3, q * dt * dt / 2},
			[]float64{q * dt * dt / 2, q * dt},
		),
		H:   FromRows([]float64{1, 0}),
		R:   FromRows([]float64{r}),
		Mu0: Vector{0, 1},
		P0:  Diag(1, 1),
	}
}

// simulateCV draws a position-velocity trajectory and the observed positions,
// returning the hidden true positions and the observations.
func simulateCV(rng *rand.Rand, m Model, t int) (truePos Vector, zs []Vector) {
	lq, _ := choleskyLower(m.Q)
	sr := math.Sqrt(m.R.At(0, 0))
	x := append(Vector(nil), m.Mu0...)
	truePos = make(Vector, t)
	zs = make([]Vector, t)
	for k := 0; k < t; k++ {
		if k > 0 {
			x = matVec(m.F, x)
			w := matVec(lq, Vector{rng.NormFloat64(), rng.NormFloat64()})
			x[0] += w[0]
			x[1] += w[1]
		}
		truePos[k] = x[0]
		zs[k] = Vector{x[0] + sr*rng.NormFloat64()}
	}
	return truePos, zs
}

// TestSmootherBeatsFilter is claim 3: the smoothed covariance has no larger
// trace than the filtered covariance at each step.
func TestSmootherBeatsFilter(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 80; trial++ {
		n := 1 + rng.Intn(3)
		mm := 1 + rng.Intn(2)
		m := randModel(rng, n, mm)
		tt := 20
		zs := simulate(rng, m, tt, nil)
		fr, err := m.Filter(zs, nil)
		if err != nil {
			t.Fatalf("trial %d filter: %v", trial, err)
		}
		sr, err := m.Smooth(zs, nil)
		if err != nil {
			t.Fatalf("trial %d smooth: %v", trial, err)
		}
		for k := 0; k < tt; k++ {
			tf := trace(fr.Filtered[k].Cov)
			ts := trace(sr.Smoothed[k].Cov)
			if ts > tf+1e-9*(1+tf) {
				t.Fatalf("trial %d step %d: smoothed trace %g > filtered %g", trial, k, ts, tf)
			}
		}
	}
}

func trace(m Matrix) float64 {
	var s float64
	for i := 0; i < m.rows; i++ {
		s += m.At(i, i)
	}
	return s
}

// TestSmootherOracle is claim 4: on a seeded constant-velocity system the
// smoother recovers the hidden position with RMSE below the measurement
// standard deviation.
func TestSmootherOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(2024))
	m := cvModel(1, 0.01, 1.0)
	tt := 300
	truePos, zs := simulateCV(rng, m, tt)
	sr, err := m.Smooth(zs, nil)
	if err != nil {
		t.Fatal(err)
	}
	var sseSmooth, sseMeas float64
	for k := 0; k < tt; k++ {
		es := sr.Smoothed[k].Mean[0] - truePos[k]
		sseSmooth += es * es
		em := zs[k][0] - truePos[k]
		sseMeas += em * em
	}
	rmseSmooth := math.Sqrt(sseSmooth / float64(tt))
	rmseMeas := math.Sqrt(sseMeas / float64(tt))
	sd := math.Sqrt(m.R.At(0, 0))
	if rmseSmooth >= sd {
		t.Fatalf("smoothed RMSE %g not below measurement sd %g", rmseSmooth, sd)
	}
	if rmseSmooth >= rmseMeas {
		t.Fatalf("smoothed RMSE %g not below raw measurement RMSE %g", rmseSmooth, rmseMeas)
	}
	t.Logf("smoothed RMSE %.4f, measurement RMSE %.4f, sd %.4f", rmseSmooth, rmseMeas, rmseMeas)
}

// TestEMMonotone is claim 5: the EM log-likelihood is non-decreasing across
// iterations, on a random system started from a plain init.
func TestEMMonotone(t *testing.T) {
	rng := rand.New(rand.NewSource(555))
	for trial := 0; trial < 30; trial++ {
		n := 1 + rng.Intn(2)
		mm := 1
		truth := randModel(rng, n, mm)
		zs := simulate(rng, truth, 120, nil)
		init := Model{
			F: Ident(n), Q: Diag(ones(n, 0.5)...), H: randMat(rng, mm, n),
			R: Diag(ones(mm, 1.0)...), Mu0: make(Vector, n), P0: Diag(ones(n, 1.0)...),
		}
		res, err := Learn(zs, init, WithMaxIterations(60))
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		for i := 1; i < len(res.LogLik); i++ {
			if res.LogLik[i] < res.LogLik[i-1]-1e-6*(1+math.Abs(res.LogLik[i-1])) {
				t.Fatalf("trial %d: loglik dropped at iter %d: %g -> %g", trial, i, res.LogLik[i-1], res.LogLik[i])
			}
		}
	}
}

func ones(n int, v float64) []float64 {
	o := make([]float64, n)
	for i := range o {
		o[i] = v
	}
	return o
}

// TestEMRecoversParams is claim 6b (see SEALED-PREDICTION.md: claim 6 as sealed,
// recovering H while it is free, is not met, because the velocity absorbs the
// state scale and H is then unidentifiable). This is the identifiable
// subproblem: in a scalar system with the transition f and the observation h
// held at the truth, the process and measurement variances q and r are
// recovered from a wrong initialization within a relative tolerance.
func TestEMRecoversParams(t *testing.T) {
	rng := rand.New(rand.NewSource(321))
	const fTrue, qTrue, rTrue = 0.7, 0.5, 1.0
	truth := Model{
		F: FromRows([]float64{fTrue}), Q: FromRows([]float64{qTrue}),
		H: FromRows([]float64{1}), R: FromRows([]float64{rTrue}),
		Mu0: Vector{0}, P0: FromRows([]float64{1}),
	}
	zs := simulateScalar(rng, truth, 20000)
	init := cloneModel(truth)
	init.Q = FromRows([]float64{0.05})
	init.R = FromRows([]float64{5})
	res, err := Learn(zs, init, WithFixed("F", "H", "Mu0", "P0"), WithMaxIterations(1000), WithTolerance(1e-10))
	if err != nil {
		t.Fatal(err)
	}
	qRec := res.Model.Q.At(0, 0)
	rRec := res.Model.R.At(0, 0)
	if d := math.Abs(qRec-qTrue) / qTrue; d > 0.1 {
		t.Fatalf("q recovered %g, want ~%g (rel err %g)", qRec, qTrue, d)
	}
	if d := math.Abs(rRec-rTrue) / rTrue; d > 0.1 {
		t.Fatalf("r recovered %g, want ~%g (rel err %g)", rRec, rTrue, d)
	}
	t.Logf("q %.4f (want %g), r %.4f (want %g), iters %d, converged %v", qRec, qTrue, rRec, rTrue, res.Iters, res.Converged)
}

// simulateScalar draws observations from a scalar linear Gaussian system.
func simulateScalar(rng *rand.Rand, m Model, t int) []Vector {
	f := m.F.At(0, 0)
	h := m.H.At(0, 0)
	sq := math.Sqrt(m.Q.At(0, 0))
	sr := math.Sqrt(m.R.At(0, 0))
	x := m.Mu0[0]
	zs := make([]Vector, t)
	for k := 0; k < t; k++ {
		if k > 0 {
			x = f*x + sq*rng.NormFloat64()
		}
		zs[k] = Vector{h*x + sr*rng.NormFloat64()}
	}
	return zs
}
