package main

import (
	"log"
	"os"
	"math/rand"
	"fmt"
)

type Neuron struct {
	Input []float64
	Output float64
	Weights []float64
	ActFn string
	Bias float64
}
func (N *Neuron) Compile_Linear() {
	op := MatMul(N.Input , N.Weights) + N.Bias
	if op >= 1 {
		N.Output = 1
	} else {
		N.Output = 0
	}
}

func (N *Neuron) Compile_RELU() {
	op := MatMul(N.Input , N.Weights) + N.Bias
	if op >= 1 {
		N.Output = op
	} else {
		N.Output = 0
	}
}

type Layer struct {
	Units int
	Neurons []*Neuron
}

type Sequential struct {
	Layers []Layer
}

func (Seq *Sequential) Add_Layer(Units int) {
	Seq.Layers = append(Seq.Layers, Layer{Units , make([]*Neuron , Units)})
}

func (Seq *Sequential) Compile() {
	for i := 1; i < len(Seq.Layers)-1; i++ {
		NIP := Seq.Layers[i].Neurons
		NOP := Seq.Layers[i+1].Neurons
		for j := 0; j < len(NOP); i++ {
			arr := []float64{}
			for i := 0; i < len(NIP); i++ {
				arr = append(arr, NIP[i].Output)
			}
			NOP[j].Input = arr
		}
	}
}

func MatMul(X, Y []float64) float64 {
	Prod := 0.0
	if len(X) != len(Y) {
		log.Fatalln("Dimension Error - X and Y not of same length")
		os.Exit(1)
	} else {
		for i := 0; i < len(X); i++ {
			Prod += X[i] * Y[i]
		}
	}
	return Prod
}

func MatAdd(X, Y []float64) []float64 {
	arr := []float64{}
	if len(X) != len(Y) {
		log.Fatalln("Dimension Error - X and Y not of same length")
		os.Exit(1)
	} else {
		for i := 0; i < len(X); i++ {
			arr = append(arr, X[i]+Y[i])
		}
	}
	return arr
}

func MatMulScale(k float64, Mat []float64) []float64 {
	arr:= make([]float64 , len(Mat))
	for i := 0; i < len(Mat); i++ {
		arr[i] = k*Mat[i]
	}
	return arr
}

func MatAddScale(k float64, Mat []float64) []float64 {
	arr:= make([]float64 , len(Mat))
	for i := 0; i < len(Mat); i++ {
		arr[i] = k + Mat[i] + rand.NormFloat64()
	}
	return arr
}
func Linspace(Low , High , Distance float64) []float64 {
	arr := []float64{}
	i := Low
	for i <= High {
		arr = append(arr, i)
		i += Distance
	}
	return arr
}

var X = Linspace(1, 100, 1)
// Y = M*X + C
var Y =	MatAddScale(4 , MatMulScale(2, X))

func main() {
	Seq := Sequential{[]Layer{}}
	Seq.Add_Layer(2)
	Seq.Add_Layer(1)
	Seq.Compile()
}

func Train_Network(Layer []*Neuron , X , Y []float64, Steps int) {
	for i := 0; i < Steps; i++ {
		for n := 0; n < len(Layer); n++ {
			if n <= 1 {
				Layer[n].Input = []float64{X[i]}
			}
			if n == 2 {
				Layer[n].Compile_Linear()
			} else {
				Layer[n].Compile_RELU()
			}
		}
		Error := Y[i] - Layer[2].Output
		BackPropagate(Error , 0.005, Layer)
	}
}

func BackPropagate(Error float64, learning_rate float64, Layer []*Neuron) {
	for i := 0; i < len(Layer); i++ {
		Layer[i].Weights = MatAdd(Layer[i].Weights ,MatMulScale(learning_rate, MatMulScale(Error , Layer[i].Input)))
	}
}
