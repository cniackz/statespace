package statespace

import (
	"errors"
	"math"
	"testing"
)

func goodModel() Model {
	return Model{
		F:   FromRows([]float64{1, 1}, []float64{0, 1}),
		Q:   Diag(1e-3, 1e-3),
		H:   FromRows([]float64{1, 0}),
		R:   FromRows([]float64{1}),
		Mu0: Vector{0, 0},
		P0:  Diag(1, 1),
	}
}

// TestValidateRejects is claim 9: every malformed model is rejected with an
// error wrapping ErrInvalidModel, and the good model is accepted.
func TestValidateRejects(t *testing.T) {
	if err := goodModel().Validate(); err != nil {
		t.Fatalf("good model rejected: %v", err)
	}
	cases := []struct {
		name string
		mut  func(*Model)
	}{
		{"F not square", func(m *Model) { m.F = Zeros(2, 3) }},
		{"Q wrong size", func(m *Model) { m.Q = Diag(1) }},
		{"H wrong columns", func(m *Model) { m.H = FromRows([]float64{1, 0, 0}) }},
		{"R wrong size", func(m *Model) { m.R = Diag(1, 1) }},
		{"P0 wrong size", func(m *Model) { m.P0 = Diag(1) }},
		{"Mu0 wrong length", func(m *Model) { m.Mu0 = Vector{0} }},
		{"F non-finite", func(m *Model) { m.F = FromRows([]float64{math.NaN(), 0}, []float64{0, 1}) }},
		{"Q non-finite", func(m *Model) { m.Q = FromRows([]float64{math.Inf(1), 0}, []float64{0, 1}) }},
		{"Mu0 non-finite", func(m *Model) { m.Mu0 = Vector{math.NaN(), 0} }},
		{"Q asymmetric", func(m *Model) { m.Q = FromRows([]float64{1, 0.5}, []float64{-0.5, 1}) }},
		{"Q not PSD", func(m *Model) { m.Q = FromRows([]float64{1, 2}, []float64{2, 1}) }},
		{"P0 not PSD", func(m *Model) { m.P0 = FromRows([]float64{-1, 0}, []float64{0, 1}) }},
		{"R not PD (zero)", func(m *Model) { m.R = FromRows([]float64{0}) }},
		{"R not PD (negative)", func(m *Model) { m.R = FromRows([]float64{-1}) }},
		{"B wrong rows", func(m *Model) { m.B = FromRows([]float64{1}) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := goodModel()
			tc.mut(&m)
			err := m.Validate()
			if err == nil {
				t.Fatalf("%s: accepted, want rejected", tc.name)
			}
			if !errors.Is(err, ErrInvalidModel) {
				t.Fatalf("%s: error %v does not wrap ErrInvalidModel", tc.name, err)
			}
		})
	}
}

// TestInputErrors covers the observation and control input checks.
func TestInputErrors(t *testing.T) {
	k, err := NewKF(goodModel())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := k.Update(Vector{1, 2}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("wrong-length z: got %v", err)
	}
	if _, err := k.Update(Vector{math.NaN()}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("non-finite z: got %v", err)
	}
	// A nil observation is a gap, not an error.
	if ll, err := k.Update(nil); err != nil || ll != 0 {
		t.Fatalf("gap update: ll %g err %v", ll, err)
	}
}

// TestGapEqualsPredictOnly is claim 8 at the online level: a gap advances the
// filter by the time update alone.
func TestGapEqualsPredictOnly(t *testing.T) {
	m := goodModel()
	a, _ := NewKF(m)
	b, _ := NewKF(m)
	// a takes a gap; b does nothing but the same predict.
	if err := a.Predict(nil); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Update(nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Predict(nil); err != nil {
		t.Fatal(err)
	}
	if d := maxVecDiff(a.State(), b.State()); d != 0 {
		t.Fatalf("gap changed the state by %g", d)
	}
	if d := maxMatDiff(a.Cov(), b.Cov()); d != 0 {
		t.Fatalf("gap changed the covariance by %g", d)
	}
}

// TestCloneIndependent checks that a clone does not share state with the
// original.
func TestCloneIndependent(t *testing.T) {
	k, _ := NewKF(goodModel())
	k.Update(Vector{1})
	c := k.Clone()
	c.Predict(nil)
	c.Update(Vector{5})
	// Mutating the clone must not move the original.
	before := k.State()
	c.Update(Vector{9})
	if d := maxVecDiff(before, k.State()); d != 0 {
		t.Fatalf("clone shares state with original (moved by %g)", d)
	}
}

// TestFilterMatchesOnlineKF checks that the batch Filter and the online KF agree
// on a simple run with a gap.
func TestFilterMatchesOnlineKF(t *testing.T) {
	m := goodModel()
	zs := []Vector{{0.1}, {0.9}, nil, {3.2}, {4.0}}
	res, err := m.Filter(zs, nil)
	if err != nil {
		t.Fatal(err)
	}
	k, _ := NewKF(m)
	var ll float64
	for step, z := range zs {
		if step > 0 {
			k.Predict(nil)
		}
		u, _ := k.Update(z)
		ll += u
		if d := maxVecDiff(res.Filtered[step].Mean, k.State()); d > 1e-12 {
			t.Fatalf("step %d: batch vs online mean diff %g", step, d)
		}
	}
	if d := math.Abs(res.LogLikelihood - ll); d > 1e-12 {
		t.Fatalf("loglik batch %g vs online %g", res.LogLikelihood, ll)
	}
}
