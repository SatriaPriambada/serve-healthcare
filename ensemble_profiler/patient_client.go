package main

import (
	"fmt"
	"flag"
	"io/ioutil"
	"net/http"
	"time"
	"math"
	"math/rand"
	"strconv"
)

var Maximum_backoff = 32000 //in millisecond

func on_disconnect(url string, backoff_counter int, ch chan<- string){
	backoff_counter++
	increasing_backoff := math.Pow(2, float64(backoff_counter))
	random_number_milliseconds := rand.Intn(1000)
	final_backoff := math.Min((increasing_backoff+float64(random_number_milliseconds)), float64(Maximum_backoff))
	fmt.Printf("backing off for the %d time: %.2f ", backoff_counter, final_backoff)
	time.Sleep(time.Duration(final_backoff) * time.Millisecond)
	// fmt.Println("finish sleeping : %.2f, now call request %s", final_backoff, url)
	MakeRequest(url, backoff_counter, ch)
}

func MakeRequest(url string, backoff_counter int, ch chan<- string) {
	// fmt.Println("start request ", url)
	http.DefaultClient.Timeout = time.Minute * 2
	start := time.Now()
	resp, err := http.Get(url)
	secs := time.Since(start).Seconds()
	if err != nil {
		fmt.Println("handle error get")
		fmt.Println("on_disconnect: ", err)
		on_disconnect(url, backoff_counter, ch)
	}
	defer resp.Body.Close()
	body, errRead := ioutil.ReadAll(resp.Body)
	if errRead != nil {
		fmt.Println("handle error read response body")
		fmt.Println(err)
	}
	ch <- fmt.Sprintf("%.2f elapsed with response length: %s %s", secs, body, url)
}

func main() {
	totalRequest:=3750
	patientId:= flag.String("patientId", "0", "string to represent a patient client id")
	flag.Parse()
	fmt.Println("patient id:", *patientId)
	start := time.Now()
	ch := make(chan string)
	for i := 0; i <= totalRequest; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(8 * time.Millisecond)
		// This how actual client will send the result
		// go MakeRequest("http://127.0.0.1:5000/hospital?patient_name=Adam&value=0.0&vtype=ECG", ch)
		// This is how profiling result is send
		// fmt.Printf("client %s alive : loop: %d ", *patientId, i)
		hostAddr := "http://127.0.0.1:5000"
		go MakeRequest( hostAddr + "/hospital?patient_id=0" +"&value="+ strconv.Itoa(i) + ".0&vtype=ECG", 0, ch)
	}
	for i := 0; i <= totalRequest; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("client finished %.2fs elapsed\n", time.Since(start).Seconds())
	// sleep 1 minute to make sure all previous request and socket file descriptor are closed
	time.Sleep(time.Minute * 1)
}
