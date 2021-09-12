import csv
import twitter


def write_to_csv(tweet):
	print(tweet["entities"].keys())
	# choose what features to have and save them in csv
	# tweet["created_at"], tweet["id"], tweet["full_text"] ...
	with open("tweets.csv", mode="a") as tweets_file:
		tweet_writer = csv.writer(tweets_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		try:
			tweet_writer.writerow([
								tweet["created_at"] if "created_at" in tweet.keys() else "None", 
								tweet["id"],
								tweet["full_text"].encode("utf-8") if "full_text" in tweet.keys() else tweet["text"].encode("utf-8"),
								tweet["truncated"] if "truncated" in tweet.keys() else "None",
								tweet["entities"]["hashtags"] if "entities" in tweet.keys() and "hashtags" in tweet["entities"].keys() else "None" ,
								tweet["entities"]["symbols"] if  "entities" in tweet.keys() and "symbols" in tweet["entities"].keys() else "None" ,
								tweet["entities"]["user_mentions"] if "entities" in tweet.keys() and "user_mentions" in tweet["entities"].keys() else "None" ,
								tweet["entities"]["urls"][0]["url"] if "entities" in tweet.keys() and "urls" in tweet["entities"].keys() and len(tweet["entities"]["urls"])>0 else "None" ,
								tweet["entities"]["urls"][0]["expanded_url"] if "entities" in tweet.keys() and "urls" in tweet["entities"].keys() and len(tweet["entities"]["urls"])>0 else "None" ,
								tweet["entities"]["urls"][0]["display_url"] if "entities" in tweet.keys() and "urls" in tweet["entities"].keys() and len(tweet["entities"]["urls"])>0 else "None" ,
								tweet["entities"]["media"][0]["id"] if "entities" in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["media_url"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["media_url_https"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["url"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["display_url"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["expanded_url"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["entities"]["media"][0]["type"] if "entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["entities"]["media"])>0 else "None" ,
								tweet["extended_entities"]["media"][0]["id"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None" ,
								tweet["extended_entities"]["media"][0]["media_url"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["media_url_https"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["url"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["display_url"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["expanded_url"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["type"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["extended_entities"]["media"][0]["additional_media_info"]["monetizable"] if "extended_entities"  in tweet.keys() and "media" in tweet["entities"].keys() and len(tweet["extended_entities"]["urls"])>0 else "None",
								tweet["source"] if "source" in tweet.keys() else "None",
								tweet["in_reply_to_status_id"] if "in_reply_to_status_id" in tweet.keys() else "None",
								tweet["in_reply_to_user_id"] if "in_reply_to_user_id" in tweet.keys() else "None",
								tweet["in_reply_to_screen_name"] if "in_reply_to_screen_name" in tweet.keys() else "None",
								tweet["user"]["id_str"] if "user" in tweet.keys() and "id" in tweet["user"].keys() else "None",
								tweet["user"]["name"] if "user" in tweet.keys() and "name" in tweet["user"].keys() else "None",
								tweet["user"]["screen_name"] if "user" in tweet.keys() and "screen_name" in tweet["user"].keys() else "None",
								tweet["user"]["location"] if "user" in tweet.keys() and "location" in tweet["user"].keys() else "None",
								tweet["user"]["description"] if "user" in tweet.keys() and "description" in tweet["user"].keys() else "None",
								tweet["user"]["url"] if "user" in tweet.keys() and "url" in tweet["user"].keys() else "None",
								tweet["user"]["entities"]["url"]["urls"][0]["url"] if "user"  in tweet.keys() and "entities" in tweet["user"].keys() and "url" in tweet["user"]["entities"].keys() and "urls" in tweet["user"]["entities"]["url"].keys() and len(tweet["user"]["entities"]["url"]["urls"])>0 else "None" ,
								tweet["user"]["entities"]["url"]["urls"][0]["expanded_url"] if "user"  in tweet.keys() and "entities" in tweet["user"].keys() and "url" in tweet["user"]["entities"].keys() and "urls" in tweet["user"]["entities"]["url"].keys() and len(tweet["user"]["entities"]["url"]["urls"])>0 else "None",
								tweet["user"]["entities"]["url"]["urls"][0]["display_url"] if "user"  in tweet.keys() and "entities" in tweet["user"].keys() and "url" in tweet["user"]["entities"].keys() and "urls" in tweet["user"]["entities"]["url"].keys() and len(tweet["user"]["entities"]["url"]["urls"])>0 else "None",
								tweet["user"]["protected"] if "user" in tweet.keys() and "protected" in tweet["user"].keys() else "None",
								tweet["user"]["followers_count"] if "user" in tweet.keys() and "followers_count" in tweet["user"].keys() else "None",
								tweet["user"]["friends_count"] if "user" in tweet.keys() and "friends_count" in tweet["user"].keys() else "None",
								tweet["user"]["listed_count"] if "user" in tweet.keys() and "listed_count" in tweet["user"].keys() else "None",
								tweet["user"]["created_at"] if "user" in tweet.keys() and "created_at" in tweet["user"].keys() else "None",
								tweet["user"]["favourites_count"] if "user" in tweet.keys() and "favourites_count" in tweet["user"].keys() else "None",
								tweet["user"]["utc_offset"] if "user" in tweet.keys() and "utc_offset" in tweet["user"].keys() else "None",
								tweet["user"]["time_zone"] if "user" in tweet.keys() and "time_zone" in tweet["user"].keys() else "None",
								tweet["user"]["geo_enabled"] if "user" in tweet.keys() and "geo_enabled" in tweet["user"].keys() else "None",
								tweet["user"]["verified"] if "user" in tweet.keys() and "verified" in tweet["user"].keys() else "None",
								tweet["user"]["statuses_count"] if "user" in tweet.keys() and "statuses_count" in tweet["user"].keys() else "None",
								tweet["user"]["lang"] if "user" in tweet.keys() and "lang" in tweet["user"].keys() else "None",
								tweet["user"]["contributors_enabled"] if "user" in tweet.keys() and "contributors_enabled" in tweet["user"].keys() else "None",
								tweet["user"]["is_translator"] if "user" in tweet.keys() and "is_translator" in tweet["user"].keys() else "None",
								tweet["user"]["is_translator_enabled"] if "user" in tweet.keys() and "is_translator_enabled" in tweet["user"].keys() else "None",
								tweet["user"]["profile_background_color"] if "user" in tweet.keys() and "profile_background_color" in tweet["user"].keys() else "None",
								tweet["user"]["profile_background_image_url"] if "user" in tweet.keys() and "profile_background_image_url" in tweet["user"].keys() else "None",
								tweet["user"]["profile_background_image_url_https"] if "user" in tweet.keys() and "profile_background_image_url_https" in tweet["user"].keys() else "None",
								tweet["user"]["profile_background_tile"] if "user" in tweet.keys() and "profile_background_tile" in tweet["user"].keys() else "None",
								tweet["user"]["profile_image_url"] if "user" in tweet.keys() and "profile_image_url" in tweet["user"].keys() else "None",
								tweet["user"]["profile_image_url_https"] if "user" in tweet.keys() and "profile_image_url_https" in tweet["user"].keys() else "None",
								tweet["user"]["profile_banner_url"] if "user" in tweet.keys() and "profile_banner_url" in tweet["user"].keys() else "None",
								tweet["user"]["profile_link_color"] if "user" in tweet.keys() and "profile_link_color" in tweet["user"].keys() else "None",
								tweet["user"]["profile_sidebar_border_color"] if "user" in tweet.keys() and "profile_sidebar_border_color" in tweet["user"].keys() else "None",
								tweet["user"]["profile_sidebar_fill_color"] if "user" in tweet.keys() and "profile_sidebar_fill_color" in tweet["user"].keys() else "None",
								tweet["user"]["profile_text_color"] if "user" in tweet.keys() and "profile_text_color" in tweet["user"].keys() else "None",
								tweet["user"]["profile_use_background_image"] if "user" in tweet.keys() and "profile_use_background_image" in tweet["user"].keys() else "None",
								tweet["user"]["has_extended_profile"] if "user" in tweet.keys() and "has_extended_profile" in tweet["user"].keys() else "None",
								tweet["user"]["default_profile"] if "user" in tweet.keys() and "default_profile" in tweet["user"].keys() else "None", 
								tweet["user"]["default_profile_image"] if "user" in tweet.keys() and "default_profile_image" in tweet["user"].keys() else "None",
								tweet["user"]["following"] if "user" in tweet.keys() and "following" in tweet["user"].keys() else "None",
								tweet["user"]["follow_request_sent"] if "user" in tweet.keys() and "follow_request_sent" in tweet["user"].keys() else "None",
								tweet["user"]["notifications"] if "user" in tweet.keys() and "notifications" in tweet["user"].keys() else "None",
								tweet["user"]["translator_type"] if "user" in tweet.keys() and "translator_type" in tweet["user"].keys() else "None",
								tweet["geo"] if "geo" in tweet.keys() else "None",
								tweet["coordinates"] if "coordinates" in tweet.keys() else "None",
								tweet["place"] if "place" in tweet.keys() else "None",
								tweet["contributors"] if "contributors" in tweet.keys() else "None",
								tweet["is_quote_status"] if "is_quote_status" in tweet.keys() else "None",
								tweet["retweet_count"] if "retweet_count" in tweet.keys() else "None",
								tweet["favorite_count"] if "favorite_count" in tweet.keys() else "None",
								tweet["favorited"] if "favorited" in tweet.keys() else "None",
								tweet["retweeted"] if "retweeted" in tweet.keys() else "None",
								tweet["possibly_sensitive"] if "possibly_sensitive" in tweet.keys() else "None",
								tweet["possibly_sensitive_appealable"] if "possibly_sensitive_appealable" in tweet.keys() else "None",
								tweet["lang"] if "lang" in tweet.keys() else "None"
								])
		except KeyError:
		 	#print(tweet["id"])
		 	pass

		except UnicodeEncodeError:
			pass


def main():
	i = 0
	identity_array = list()
	for x in range(2, 491):
		file_name = "json_tweets_" + str(x) + ".json"
		data = twitter.load_json(file_name)
		for tweet in data['tweets']:
			i = i + 1
			print(i)
			if tweet["id"] not in identity_array:
				write_to_csv(tweet)
				identity_array.append(tweet["id"])
			else:
				#print(tweet["id"])
				pass

	print(len(identity_array))


if __name__ == "__main__":
	main()



	
	

