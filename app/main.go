package main

// calculate execution time of fft
// need to install: go get github.com/mjibson/go-dsp
// source of code: https://pkg.go.dev/github.com/mjibson/go-dsp/fft#FFT

import (
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	prometheusMiddleware "github.com/iris-contrib/middleware/prometheus"
	"github.com/kataras/iris/v12"
	"github.com/mjibson/go-dsp/fft"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Data struct {
	a1  float64 // 第一个函数的振幅
	w1  float64 // 第一个函数的频率
	a11 float64 // 第一个函数的补充数的振幅
	w11 float64 // 第一个函数的补充数的频率
	a2  float64 // 第一个函数的振幅
	w2  float64 // 第一个函数的频率
	a21 float64 // 第一个函数的补充数的振幅
	w21 float64 // 第一个函数的补充数的频率
}

// 给原数据增加高斯噪声，防止读取缓存
func (d *Data) change2param(mu, sigma float64) {
	d.a1 = paramNormNoise(d.a1, mu, sigma)
	d.w1 = paramNormNoise(d.w1, mu, sigma)
	d.a11 = paramNormNoise(d.a11, mu, sigma)
	d.w11 = paramNormNoise(d.w11, mu, sigma)
	d.a2 = paramNormNoise(d.a2, mu, sigma)
	d.w2 = paramNormNoise(d.w2, mu, sigma)
	d.a21 = paramNormNoise(d.a21, mu, sigma)
	d.w21 = paramNormNoise(d.w21, mu, sigma)
}

func paramNormNoise(param, mu, sigma float64) float64 {
	sample := rand.NormFloat64()*sigma + mu
	param += sample
	return param
}

func calculate_fft(data Data, numSamples int, freq float64) time.Duration {
	start := time.Now()
	// Equation 3-10.
	x := func(n int, freq float64) float64 {
		wave0 := 10 * math.Sin(2.0*math.Pi*float64(n)*freq/8.0)
		wave1 := 0.5 * math.Sin(2.0*math.Pi*float64(n)*freq/4.0+3.0*math.Pi/4.0)
		return wave0 + wave1
	}

	// Discretize our function by sampling at 8 points.
	a := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		a[i] = x(i, freq)
	}

	_ = fft.FFTReal(a)
	return time.Since(start)
}

func main() {
	defaultArgs := Data{
		a1:  10,
		w1:  8,
		a11: 0,
		w11: 5,
		a2:  0.5,
		w2:  4.0,
		a21: 3.0,
		w21: 8.0,
	}
	app := iris.New()
	m := prometheusMiddleware.New("serviceName", 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
	app.Use(m.ServeHTTP)
	app.OnErrorCode(iris.StatusNotFound, func(ctx iris.Context) {
		m.ServeHTTP(ctx)
		ctx.Writef("Not Found")
	})
	app.Get("/metrics", iris.FromStd(promhttp.Handler()))

	app.Post("/fft/real/{numSamples}", func(ctx iris.Context) {
		fmt.Println("fft_real")
		numSampleStr := ctx.Params().Get("numSamples")
		numSample, err := strconv.ParseInt(numSampleStr, 10, 64)
		if err != nil {
			ctx.StatusCode(iris.StatusInternalServerError)
		}
		ctx.Writef("Length is %d", numSample)

		currentArgs := defaultArgs
		currentArgs.change2param(1.0, 5.0)
		elapseTime := calculate_fft(currentArgs, int(numSample), 0.01)
		fmt.Println(elapseTime)
	})
	app.Post("/fft", func(ctx iris.Context) {
		fmt.Println("fft")
		numSamplesStr := ctx.URLParamDefault("numSamples", "10000")
		freqStr := ctx.URLParamDefault("freq", "0.1") // shortcut for ctx.Request().URL.Query().Get("lastname")

		numSample, err := strconv.ParseInt(numSamplesStr, 10, 64)
		if err != nil {
			ctx.StatusCode(iris.StatusBadRequest)
			return
		}
		freq, err := strconv.ParseFloat(freqStr, 32)
		if err != nil {
			ctx.StatusCode(iris.StatusBadRequest)
			return
		}
		currentArgs := defaultArgs
		currentArgs.change2param(1.0, 5.0)
		elapseTime := calculate_fft(currentArgs, int(numSample), freq)
		fmt.Println(numSample, freq)
		fmt.Println(elapseTime)

		ctx.Writef("Hello %s %s", numSamplesStr, freqStr)
	})
	app.Build()

	srv := &http.Server{
		Addr:           ":8080",
		Handler:        app,
		ReadTimeout:    1 * time.Second,
		WriteTimeout:   1 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	srv.ListenAndServe()

	//app.Listen(":8080")
}
