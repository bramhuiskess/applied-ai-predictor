#!/usr/bin/env bash
# Stakeholder demo against a running backend (default http://127.0.0.1:8000).
#   docker run --rm -p 8000:8000 --network bridge applied-ai-predictor   # or: predictor serve
#   ./scripts/demo.sh
set -euo pipefail
URL="${URL:-http://127.0.0.1:8000}"
post() { curl -sS -X POST "$URL$1" -H 'content-type: application/json' -d "$2"; echo; }
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "1. Four use cases, one backend"
curl -sS "$URL/use-cases"; echo

step "2. Greenhouse: temperature in one hour (regression, time series)"
post /models/greenhouse-climate/predict '{"records":[{"record_id":"kas-3 13:00","features":{
  "hour_of_day":13,"outside_temp_c":9.5,"solar_radiation_wm2":420,"inside_humidity_pct":80,
  "co2_ppm":520,"ventilation_on":0,"heating_on":0,"inside_temp_c":21.4}}]}'

step "3. Machine failure (classification, real AI4I 2020 data)"
post /models/machine-failure/predict '{"records":[
  {"record_id":"machine-ok","features":{"product_quality":"M","air_temp_k":299.5,"process_temp_k":309.8,
   "rotational_speed_rpm":1500,"torque_nm":40,"tool_wear_min":30}},
  {"record_id":"machine-worn","features":{"product_quality":"L","air_temp_k":302.1,"process_temp_k":311.0,
   "rotational_speed_rpm":1290,"torque_nm":68,"tool_wear_min":215}}]}'

step "4. Same machines, no labels needed (anomaly detection)"
post /models/machine-anomaly/predict '{"records":[
  {"record_id":"machine-ok","features":{"product_quality":"M","air_temp_k":299.5,"process_temp_k":309.8,
   "rotational_speed_rpm":1500,"torque_nm":40,"tool_wear_min":30}},
  {"record_id":"machine-odd","features":{"product_quality":"L","air_temp_k":304.5,"process_temp_k":313.5,
   "rotational_speed_rpm":2800,"torque_nm":8,"tool_wear_min":240}}]}'

step "5. Sickness absence (aggregated, synthetic)"
post /models/absence-forecast/predict '{"records":[{"record_id":"care-2026-01","features":{
  "department_type":"care","headcount":30,"avg_age":44,"overtime_hours_per_fte":6,
  "workload_index":1.1,"flu_season":1,"previous_month_absence_pct":7.2}}]}'

step "6. Privacy policy: health data and tiny groups are refused"
post /models/absence-forecast/predict '{"records":[{"record_id":"care-2026-01","features":{
  "department_type":"care","headcount":3,"avg_age":44,"overtime_hours_per_fte":6,
  "workload_index":1.1,"flu_season":1,"previous_month_absence_pct":7.2,"diagnosis":"flu"}}]}'

step "7. Validation: clear errors instead of silent garbage"
post /models/greenhouse-climate/predict '{"records":[{"features":{
  "hour_of_day":13,"outside_temp_c":"warm","solar_radiation_wm2":420,"inside_humidity_pct":80,
  "co2_ppm":520,"ventilation_on":2,"heating_on":0,"colour":"red"}}]}'

step "8. Batch: JSONL in, JSONL out, one bad line does not stop the batch"
printf '%s\n' \
  '{"record_id":"a","features":{"product_quality":"H","air_temp_k":300,"process_temp_k":310,"rotational_speed_rpm":1400,"torque_nm":45,"tool_wear_min":100}}' \
  'not json' \
  '{"record_id":"c","features":{"product_quality":"L","air_temp_k":303,"process_temp_k":312,"rotational_speed_rpm":1250,"torque_nm":70,"tool_wear_min":220}}' \
| curl -sS -X POST "$URL/models/machine-failure/predict/batch" -H 'content-type: application/x-ndjson' --data-binary @-

step "9. Monitoring"
curl -sS "$URL/metrics"; echo
