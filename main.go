package main

import "gonum.org/v1/gonum/mat"

type Layer struct {
	// dimensions: r rows, c1 columns/features
	Input *mat.Dense
	// dimensions: c1 rows, c2 columns
	Weights *mat.Dense
	// dimensions: r rows, c2 columns
	Output *mat.Dense
}

func (l *Layer) Forward() {
	examples, _ := l.Input.Dims()
	_, outputNodes := l.Weights.Dims()
	l.Output = mat.NewDense(examples, outputNodes, nil)
	l.Output.Mul(l.Input, l.Weights)
}

type Network struct {
	Input  *mat.Dense
	Layers []*Layer
	Output *mat.Dense
}

func (n *Network) Forward() {
	if len(n.Layers) == 0 {
		return
	}
	for i, layer := range n.Layers {
		if i == 0 {
			layer.Input = n.Input
		} else {
			layer.Input = n.Layers[i-1].Output
		}
		layer.Forward()
	}
	n.Output = n.Layers[len(n.Layers)-1].Output
}
