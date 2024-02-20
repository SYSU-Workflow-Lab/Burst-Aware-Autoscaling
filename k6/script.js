import http from 'k6/http';
import { sleep } from 'k6';

export function randomNumGen() {
  var u = Math.random();
  var mu = 1;
  return -Math.log(1.0 - u) / mu;
}


export const options = {
    discardResponseBodies: true,
    scenarios: {
      contacts: {
        executor: 'externally-controlled', // 这个不能改变
        vus: 5, // 开始的VU
        maxVUs: 10, // 最大的VU
        duration: '72h', //持续时间
      },
    },
  };

export let TIMEOUT = 666;
// export let hostname = '10.186.117.4:30001'
export let hostname = '127.0.0.1:30001' // 直接访问本地，用istio系统接过去
// export let hostname = '192.168.0.97:30001'

export default function () {
  if(__VU%2 == 1){
    sleep(Math.random()*(2-2 * TIMEOUT/1000));
    const params = {
      timeout: TIMEOUT
    }
    var startDate = Math.round(new Date().getTime());
    http.post('http://' + hostname + '/fft/real/2000', {}, params);
    var endDate = Math.round(new Date().getTime());
    // 补足400ms
    if(endDate - startDate < TIMEOUT){
      sleep((TIMEOUT - endDate + startDate)/1000);
    } 
  }else{
    const params = {
      timeout: TIMEOUT
    }
    var startDate = Math.round(new Date().getTime());
    http.post('http://' + hostname + '/fft/real/2000', {}, params); // 当时测得fft 2000，响应时间比较小，能够打到较大的流量
    var endDate = Math.round(new Date().getTime());
    // 补足400ms
    if(endDate - startDate < TIMEOUT){
      sleep((TIMEOUT - endDate + startDate)/1000);
    } 
    sleep(Math.random()*(2-2 * TIMEOUT/1000));
  }
    // 下面均为有偏向
    // sleep(Math.random()*2); // constUserPecSec
    // sleep(randomNumGen()); // poisson distribution
    // sleep(1); // atOnceUser
}
