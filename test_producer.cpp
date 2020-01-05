#include <cppkafka/cppkafka.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <regex>

using namespace cppkafka;
using json = nlohmann::json;

int main() {

  // load json file and prerocess it to be read by the json package
  std::ifstream tweets_file("./Tweets/10-min/test.json", std::ifstream::binary);
  std::string tweets_str((std::istreambuf_iterator<char>(tweets_file)),
                 std::istreambuf_iterator<char>());
  std::regex regexp("(\\})(\n\\{\"created_at\":)");
  std::regex regexpdel("\\{\"delete\":.*\\}\n");
  tweets_str = std::regex_replace(tweets_str, regexpdel, "");
  tweets_str = std::regex_replace(tweets_str, regexp, "},$2");
  tweets_str.insert(tweets_str.begin(), '[');
  tweets_str.insert(tweets_str.end(), ']');

  json tweets = json::parse(tweets_str);

  // Create the config for kafka
  Configuration config = {
      { "metadata.broker.list", "127.0.0.1:9092" }
  };
  // Create the Producer
  Producer producer(config);

  for (auto& tweet : tweets) {
    std::string tweet_text = tweet["text"];
    producer.produce(MessageBuilder("tweets_stream").partition(0).payload(tweet_text));
  }
  producer.flush();

  return 0;
}
