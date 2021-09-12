import tweepy
import boto3
import json


def write_to_json(data, file_name):
	with open(file_name, 'w', encoding='utf8') as outfile:
		json.dump(data, outfile, indent=4)


def load_json(file_name):
	with open(file_name, 'r+') as json_updated:
		data = json.load(json_updated)
	return data


def since_id_read(filename):
	with open(filename, 'r') as outfile:
		id = outfile.read()
	return id


def since_id_write(filename, id):
	with open(filename, 'w') as outfile:
		outfile.write(id)


def start_script(i):
	consumer_key = "5tatATeyTSXGSJmAhmUoTV0y3"
	consumer_secret_key = "D03B203obCUAvU22RYHVkzxfF6kxsMUowic5KSkufTA4GdsUl1"
	access_token = "1238829219179880449-uDM1UR2elyF9XqEtdSutgbV08bxcS1"
	access_token_secret = "eCtMcrGZloGIn1yRiRygLVSiD4SMhLAiLV2axlYfyNn10"

	s3_client = boto3.resource('s3')

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
	# Setting your access token and secret
	auth.set_access_token(access_token, access_token_secret)
	# Creating the API object while passing in auth information
	api = tweepy.API(auth)

	since_id = since_id_read('since_id.txt')
	print(since_id, flush=True)

	public_tweets = api.home_timeline(tweet_mode='extended', count=200, since_id=int(since_id))
	#
	since_id_new = public_tweets[0]._json['id']
	print(len(public_tweets), flush=True)
	print(since_id_new, flush=True)
	since_id_write('since_id.txt', str(since_id_new))
	#print(public_tweets[0])

	file = "json_tweets_" + str(i) + ".json"
	for tweet in public_tweets:
		old_data = load_json(file)
		temp = old_data['tweets']
		temp.append(tweet._json)
		write_to_json(old_data, file)


	
# foreach through all tweets pulled
#for tweet in public_tweets:
   # printing the text stored inside the tweet object
 #  print(tweet.full_text.encode('utf8'))
  # print(tweet.user)