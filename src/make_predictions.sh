#!/usr/bin/env sh

echo "Predict Celius for Fahrenheit temperature [32 212 200 100 150]"
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{"data": [32, 212, 200, 100, 150]}'
echo ""
echo "Done!"
