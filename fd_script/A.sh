export https_proxy=
export http_proxy=

curl -X POST "http://10.174.148.79:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "李彦宏是谁呢？"},
  ],
  "max_tokens": 200,
  "min_tokens": 200,
  "top_p": 0
}'


