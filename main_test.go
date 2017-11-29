package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSingleLayerNetwork(t *testing.T) {
	// Network setup: input (3 nodes), hidden layer (2 nodes), output (1 node)
	inputs := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	w0 := mat.NewDense(3, 2, []float64{1, 1, 1, 1, 1, 1})
	w1 := mat.NewDense(2, 1, []float64{1, 1})
	n := &Network{
		Input: inputs,
		Layers: []*Layer{
			&Layer{Weights: w0},
			&Layer{Weights: w1},
		},
	}
	n.Forward()
	if n.Output.At(0, 0) != 12.0 {
		t.Error("Network should calculate double the sum of the inputs")
	}
	if n.Output.At(1, 0) != 30.0 {
		t.Error("Network should calculate double the sum of the inputs")
	}
}
