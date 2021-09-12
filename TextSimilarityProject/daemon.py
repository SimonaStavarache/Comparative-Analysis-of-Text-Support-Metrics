import twitter
import time

i = 345
while True:
	print("Creating new file", flush=True)
	json_start = "{\n\"tweets\":[\n]}"
	filename = "json_tweets_" + str(i) + ".json"
	with open(filename, "w+") as outfile:
		outfile.write(json_start)
	twitter.start_script(i)
	print("Script finished", flush=True)
	i += 1
	print("Waiting....", flush=True)
	time.sleep(3600)
