package main

import (
	"fmt"
	"flag"
	"io/ioutil"
	"net/http"
	"time"
	"math"
	"math/rand"
)

var Exp_backoff_counter = 0
var Maximum_backoff = 32000 //in millisecond

func on_disconnect(url string, ch chan<- string){
	Exp_backoff_counter++
	const_backoff := math.Pow(2, float64(Exp_backoff_counter))
	random_number_milliseconds := rand.Intn(Maximum_backoff)
	final_backoff := math.Min((const_backoff+float64(random_number_milliseconds)), float64(Maximum_backoff))
	fmt.Printf("backing off : %.2f ", final_backoff)
	time.Sleep(time.Duration(final_backoff) * time.Millisecond)
	go MakeRequest(url, ch)
}
	
func MakeRequest(url string, ch chan<- string) {
	start := time.Now()
	resp, err := http.Get(url)
	secs := time.Since(start).Seconds()
	if err != nil {
		fmt.Println("handle error get")
		fmt.Println("on_disconnect: ", err)
		on_disconnect(url, ch)
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
	patientId:= flag.String("patientId", "0", "string to represent a patient client id")
	flag.Parse()
	fmt.Println("patient id:", *patientId)
	start := time.Now()
	ch := make(chan string)
	for i := 0; i <= 3800; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(8 * time.Millisecond)
		// This how actual client will send the result
		// go MakeRequest("http://127.0.0.1:5000/hospital?patient_name=Adam&value=0.0&vtype=ECG", ch)
		// This is how profiling result is send
		fmt.Printf("client alive %s : loop: %d ", *patientId, i)
		hostAddr := "http://127.0.0.1:5000"
		go MakeRequest( hostAddr + "/hospital?patient_id=0" +"&value=0.0&vtype=ECG", ch)
	}
	for i := 0; i <= 3800; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
