import { sleep } from 'k6'
import http from 'k6/http'

// See https://k6.io/docs/using-k6/options
export const options = {
//   stages: [
//     { duration: "1m", target: 10 },
//     { duration: "3m", target: 10 },
//     { duration: "1m", target: 0 },
//   ],
  scenarios: {
    contacts: {
      executor: 'externally-controlled',
      vus: 20,
      duration: "5m",
      maxVUs: 30,
    }
  },  
  ext: {
    loadimpact: {
      distribution: {
        'amazon:us:ashburn': { loadZone: 'amazon:us:ashburn', percent: 100 },
      },
    },
  },
}

export default function main() {
  let response = http.get('https://test-api.k6.io/public/crocodiles/')
  sleep(1);
}