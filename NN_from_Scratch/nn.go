package main

import (
	"log"
	"os"
	"math/rand"
	"time"
	"strconv"
)

var GlobalRandSeed int64 = 101

type Neuron struct {
	Input []float64
	Output float64
	Weights []float64
	ActFn string
	Bias float64
}
func (N *Neuron) Compile_Linear() {
	N.Output = MatMul(N.Input , N.Weights) + N.Bias
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

func (Seq *Sequential) Add_Layer(Input_dims , Units int , ActFn string) {
	NN := make([]*Neuron , Units)
	for n := 0; n < Units; n++ {
		NN[n] = &Neuron{make([]float64, Input_dims), 0.0 , []float64{}, ActFn, 0}
	}
	Seq.Layers = append(Seq.Layers, Layer{Units , NN})
}

func (Seq *Sequential) Compile() {
	for i := 1; i < len(Seq.Layers)-1; i++ {
		NIP := Seq.Layers[i].Neurons
		NOP := Seq.Layers[i+1].Neurons
		for j := 0; j < len(NOP); i++ {
			arr := []float64{}
			wArr := []float64{}
			for i := 0; i < len(NIP); i++ {
				arr = append(arr, NIP[i].Output)
				Seeder()
				wArr = append(wArr, rand.Float64())
			}
			NOP[j].Input = arr
			NOP[j].Input = wArr
		}
	}
//	For Output Layer
	Layers := Seq.Layers
	NIP := Layers[len(Layers)-2].Neurons
	NOP := Layers[len(Layers)-1].Neurons
	arr := []float64{}
	wArr := []float64{}
	for i := 0; i < len(NIP); i++ {
		arr = append(arr, NIP[i].Output)
		Seeder()
		wArr = append(wArr, rand.Float64())
	}
	NOP[0].Input = arr
	NOP[0].Weights = wArr
}
func (Seq * Sequential) BackPropagate(Error , Learning_Rate float64) {
	Layers := Seq.Layers
	for i := 0; i < len(Layers);i++ {
		NN := Layers[i].Neurons
		for n := 0; n < len(NN); i++ {
			NN[n].Weights = MatAdd(NN[n].Weights, MatMulScale(Error , MatMulScale(Learning_Rate ,NN[n].Input)))
		}
	}
}

func (Seq *Sequential) Train(X, Y []float64, Steps int) {
	Layers := Seq.Layers
	for i := 0; i < len(X); i++ {
		Input_Layer := Layers[0]
		for j:= 0; j < len(Input_Layer.Neurons); j++ {
			Input_Layer.Neurons[j].Input = []float64{X[i]}
			Seeder()
			Input_Layer.Neurons[j].Weights = []float64{rand.Float64()}
		}
		for k := 1; k < len(Layers); k++ {
			NN := Layers[k].Neurons
			for n := 0; n < len(NN); n++ {
				if NN[n].ActFn == "linear" {
					NN[n].Compile_Linear()
				} else {
					NN[n].Compile_RELU()
				}
			}
		}
		Error := Y[i] - Seq.Layers[len(Seq.Layers) - 1].Neurons[0].Output
		Seq.BackPropagate(Error , 0.005)
	}
}

func (Seq *Sequential) Predict(X float64) float64 {
	Layers := Seq.Layers
	NN := Layers[0].Neurons
	for i := 0; i < len(NN); i++ {
		NN[i].Input = []float64{X}
	}
	for i := 0; i < len(Layers); i++ {
		NN := Layers[i].Neurons
		for n := 0; n < len(NN); n++ {
			if NN[n].ActFn == "linear" {
				NN[n].Compile_Linear()
			} else {
				NN[n].Compile_RELU()
			}
		}
	}
	return Layers[len(Layers) - 1].Neurons[0].Output
}

// Numpy Stuff coded from scratch (Not in a General form though)
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

func Seeder() {
	seed_val, _ := strconv.Atoi(time.Now().Format(".000")[1:])
	rand.Seed(int64(seed_val)*GlobalRandSeed)
	GlobalRandSeed += 120
}


var X = Linspace(1, 100, 1)
// Y = M*X + C
var Y =	MatAddScale(4 , MatMulScale(2, X)) //M = 2 and B = 4

func main() {
	Seq := Sequential{[]Layer{}}
	Seq.Add_Layer(1, 2 , "relu")
	Seq.Add_Layer(2, 1 , "linear")
	Seq.Compile()
	Seq.Train(X, Y , 100) // Problem is Here 
	//res := Seq.Predict(35.5)
	//fmt.Println(res)
}
