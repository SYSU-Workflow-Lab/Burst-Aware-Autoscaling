import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { sleep } from 'k6';

// const data = new SharedArray('some data name', function () {
//   return JSON.parse(open('./test.json')).users;
// });

// export let s1 = new Array()
// for(var i=0;i<data.length;i++){
//   s1.push({ duration: '60s', target: parseInt(data[i].password) })  
// }

// 静态
// export const options = {
//   discardResponseBodies: true,
//   vus: 700,
//   duration: "3m"
// };

// 线性
// export const options = {
//   discardResponseBodies: true,
//   startVUs: 0,
//   stages: [
//     { duration: '5m', target: 600 },
//   ],
// };


// 外部控制
export const options = {
  discardResponseBodies: true,
  scenarios: {
    contacts: {
      executor: 'externally-controlled',
      vus: 450,
      maxVUs: 800,
      duration: '3m',
    },
  },
};


// 当前的模式是偏向于atOnceUser的，如何让到达的用户服从泊松流
// 
// 目前的执行顺序是init -> setup -> vu -> teardown
// 有一个想法：能否在setup这边通过系统时间来进行控制，即随机指定下一个时间
// 然后VU直到该时间才能执行，或者在setup睡眠指定大小的时间等
// export function setup() {
// //   // 2. setup code
//   var myDate = new Date();
//   console.log(__VU,myDate.getSeconds(), myDate.getMilliseconds() )
// }

export function randomNumGen() {
  var u = Math.random();
  var mu = 1;
  return -Math.log(1.0 - u) / mu;
}

export let TIMEOUT = 660;
export let hostname = '10.186.117.4:30001'
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
    http.post('http://' + hostname + '/fft/real/2000', {}, params);
    var endDate = Math.round(new Date().getTime());
    // 补足400ms
    if(endDate - startDate < TIMEOUT){
      sleep((TIMEOUT - endDate + startDate)/1000);
    } 
    sleep(Math.random()*(2-2 * TIMEOUT/1000));
  }

  // 获取截止时间
  // 先睡到指定时间
  // 再睡一个间隔时间
  // sleep(Math.random()*2) // constUserPecSec
  // sleep(randomNumGen()) // poisson distribution
}