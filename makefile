init:
	export FLASK_APP=minitwit
	export NUM_SENSORS=5
db:
	flask initdb
start:
	PYTHONPATH=. flask run --host=0.0.0.0

