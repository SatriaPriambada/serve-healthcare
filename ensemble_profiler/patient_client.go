package main

import (
	"fmt"
	"flag"
	"io/ioutil"
	"net/http"
	"time"
)

func MakeRequest(url string, ch chan<- string) {
	start := time.Now()
	resp, err := http.Get(url)
	if err != nil {
		fmt.Printf("handle error")
		fmt.Println(err)
	}
	secs := time.Since(start).Seconds()
	body, _ := ioutil.ReadAll(resp.Body)
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
		// fmt.Printf("%d ", i)
		hostAddr := "http://127.0.0.1:5000"
		go MakeRequest( hostAddr + "/hospital?patient_id=0" +"&value=0.0&vtype=ECG", ch)
	}
	for i := 0; i <= 3800; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
