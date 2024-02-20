nohup k6 run script.js -a '0.0.0.0:6565' >/dev/null 2>&1 &
# The -a means address of API server, 0.0.0.0 is necessary, or server can't be accessed outside.